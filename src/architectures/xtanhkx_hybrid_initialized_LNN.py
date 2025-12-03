import torch
from torch import ones, zeros
import torch.nn as nn
from torch.nn.functional import tanh
from torch.nn.parameter import Parameter
from torch.func import vmap, hessian, jacrev, functional_call
import os
import time
import shutil
from file_exports import save_model, save_history, load_model

class chadWrapper(nn.Module):
    def __init__(self, f):
        super(chadWrapper, self).__init__()
        self.f = f()

    def forward(self, x):
        return self.f(x)

class DyT(nn.Module):
    def __init__(self, C, init_α=0.5):
        super().__init__()
        self.α = Parameter(ones(1) * init_α)
        self.γ = Parameter(ones(C))
        self.β = Parameter(zeros(C))
    def forward(self, x):
        x = tanh(self.alpha * x)
        return self.γ * x + self.β
        
class QuadraticSmoothActivation(nn.Module):
    def __init__(self, k=0.5):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x * torch.tanh(self.k * x)

class QuadraticActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x**2

class QuadraticMLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_layers: int,
                 hidden_dim: int,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.input_size = input_size
        self.dtype = dtype
        
        # Build the Quadratic MLP part
        layers = []
        in_feat = input_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_feat, hidden_dim//2))
            layers.append(QuadraticSmoothActivation())
            in_feat = hidden_dim//2
        layers.append(nn.Linear(hidden_dim//2, 1, bias=False))
        self.qmlp = nn.Sequential(*layers).to(self.device, self.dtype)
        self.to(self.device, self.dtype)

    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)                # [input_dim] → [1, input_dim]
        # 2) network output
        qmlp_out = self.qmlp(x)               # [B, 1]
        return qmlp_out


class CombinedNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_layers: int,
                 activation_fn: nn.Module,
                 hidden_dim: int,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.input_size = input_size
        self.n = input_size // 2

        # Build the MLP part
        layers = []
        in_feat = input_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_feat, hidden_dim//2))
            layers.append(nn.Softplus())
            in_feat = hidden_dim//2
        layers.append(nn.Linear(hidden_dim//2, 1, bias=False))
        self.mlp = nn.Sequential(*layers).to(self.device, dtype)
        self.qmlp = QuadraticMLP(input_size, num_layers, hidden_dim, device)
        # Build the quadratic term
        # using the same self.n so we know where q̇ starts
        self.a = 1.0

        self.to(self.device, self.dtype)

    def forward(self, x: torch.Tensor):
        # 1) ensure batch-dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)               # [input_dim] → [1, input_dim]
        # 2) network output
        mlp_out = self.mlp(x)                # [B, 1]
        qmlp_out = self.qmlp(x)
        qdot = x[:, self.n:]                 # [B, input_dim/2]
        quad_out = self.a * (qdot**2).sum(dim=1, keepdim=True)
        out = mlp_out + qmlp_out + quad_out  # [B,1]
        return out.view(-1)  # safe flatten to [B]


class LNN(nn.Module):
    def __init__(self, input_size, num_layers, activation_fn, hidden_dim, device, dtype=torch.float32):
        super(LNN, self).__init__()
        self.dtype = dtype
        if (device != None):
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.n = input_size // 2
#        self.dyt_layer = DyT(input_size).to(self.device, dtype)
            
        self.network = CombinedNetwork(
            input_size, num_layers, activation_fn, hidden_dim, device, dtype
        ).to(self.device, self.dtype)
    
    def compute_loss(self, pred, target, inputs, lambda_penalty=1.0):
        """
        Compute the total loss including both prediction loss and regularization penalty.
        
        Args:
            pred: Predicted qdot_qdotdot from forward pass
            target: Target qdot_qdotdot 
            inputs: Input q_qdot used to compute regularization
            lambda_penalty: Weight for the eigenvalue penalty term
            
        Returns:
            total_loss: Combined prediction loss + penalty
            loss_components: Dict with breakdown of loss components
        """
        # Base prediction loss
        base_loss = torch.mean(torch.abs(pred - target))
        
        # Regularization penalty (eigenvalue constraint)
        jac, hess = self.batch_jac_and_hes(self.network, inputs)
        A = hess[:, self.n:, self.n:]  # Mass matrix
        eigenvalues = torch.vmap(torch.linalg.eigvalsh)(A)
        penalty = torch.mean(torch.sum(torch.relu(-eigenvalues), dim=1))
        
        # Total loss
        total_loss = base_loss + lambda_penalty * penalty
        
        # Return components for logging/debugging
        loss_components = {
            'base_loss': base_loss.item(),
            'penalty': penalty.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

    def lagrangian(self, x):
        if (len(x.shape) < 2):
            x = x.unsqueeze(0)

#        dyt_x = self.dyt_layer(x)
#        return self.network(dyt_x)
        return self.network(x)

    def batch_jac_and_hes(self, model, inputs):
        params = dict(model.named_parameters())
    
        def func(params, x):
            return functional_call(model, params, x)
    
        def single_jac(x):
            return jacrev(func, argnums=1)(params, x)
    
        # Compute Jacobian
        jac = vmap(single_jac)(inputs)
        jac = jac.squeeze(1)
    
        # Compute Hessian
        def single_hes(x):
            return jacrev(single_jac)(x)
    
        hes = vmap(single_hes)(inputs)
        hes = hes.squeeze(1)
        
        return jac, hes

    def forward(self, x, t=0):
        if x.dim() < 2:
            x = x.unsqueeze(0)
        B, D = x.shape
        n = self.n
        qdot = x[:, n:]
    
        jac, hess = self.batch_jac_and_hes(self.network, x)
    
        A = hess[:, n:, n:]

        # --- DIAGNOSTIC ---
        # Calculate the condition number. High values mean it's close to singular.
        cond_nums = torch.linalg.cond(A)  # Returns tensor of shape [B]
        max_cond = cond_nums.max().item()
        if max_cond > 1e6:
            problematic_indices = (cond_nums > 1e6).nonzero(as_tuple=True)[0]
            print(f"WARNING: High condition numbers detected. Max: {max_cond:.2e}")
            print(f"Problematic batch indices: {problematic_indices.tolist()}")
        # ------------------
    
        Bmat = hess[:, n:, :n]
        C = jac[:, :n].unsqueeze(-1)
        q = qdot.unsqueeze(-1)
        rhs = C - torch.matmul(Bmat, q)
    
        try:
            Ap = torch.linalg.pinv(A)
        except torch.linalg.LinAlgError as e:
            cond_num = torch.linalg.cond(A).item()
            print(f"FATAL: LinAlgError during pinv! Condition number was {cond_num:.2e}. Error: {e}")
            # Return a zero tensor to prevent crashing the whole script,
            # but the trajectory will be wrong.
            return torch.zeros_like(x)
    
        qdd = torch.matmul(Ap, rhs).squeeze(-1)
        out = torch.cat([qdot, qdd], dim=1)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"FATAL: NaN/Inf in output! Condition number was {cond_num:.2e}. Input was {x.detach().cpu().numpy()}")
    
        if self.training:
            out.retain_grad()
    
        return out
    
    def fit(self, num_epochs_to_run, train_loader, optimizer, scheduler=None, 
            val_loader=None, gradient_clip_norm=1.0, ckpt_path='/kaggle/working/', lambda_penalty=1.0):
        
        history = {'train_loss': [], 'val_loss': [], 'lr': [], 
                   'train_base_loss': [], 'train_penalty': [], 'val_base_loss': [], 'val_penalty': []}
        if val_loader is None:
            # Remove validation-specific keys if no validation
            del history['val_loss'], history['val_base_loss'], history['val_penalty']
            
        print(f"Starting training for {num_epochs_to_run} epochs...")
        initial_lr_display = optimizer.param_groups[0]['lr']
        print(f"Initial LR for this run: {initial_lr_display:.2e}, Batch Size: {train_loader.batch_size}")
        if val_loader is not None:
            print(f"Validation enabled with {len(val_loader.dataset)} samples")
        print(f"Regularization penalty weight: {lambda_penalty}")
        
        best_loss = float('inf')
        best_epoch = -1
        
        # Create temporary checkpoint directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_checkpoint_dir = os.path.join(ckpt_path, f"temp_checkpoint_{timestamp}")
        os.makedirs(temp_checkpoint_dir, exist_ok=True)

        for epoch in range(1, num_epochs_to_run + 1):
            self.train() # Set model to training mode
            epoch_train_loss = 0.0
            epoch_train_base_loss = 0.0
            epoch_train_penalty = 0.0
            
            for batch_idx, (xi, yi) in enumerate(train_loader):
                # Move data to the same device as the model
                xi, yi = xi.to(self.device, self.dtype), yi.to(self.device, self.dtype)

                optimizer.zero_grad()
                qdot_qdotdot_pred = self(xi) 
                
                # Use unified loss computation
                total_loss, loss_components = self.compute_loss(qdot_qdotdot_pred, yi, xi, lambda_penalty)
                total_loss.backward()
                
                if gradient_clip_norm is not None and gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
                
                optimizer.step()
                epoch_train_loss += loss_components['total_loss']
                epoch_train_base_loss += loss_components['base_loss']
                epoch_train_penalty += loss_components['penalty']
            
            avg_epoch_train_loss = epoch_train_loss / len(train_loader)
            avg_train_base_loss = epoch_train_base_loss / len(train_loader)
            avg_train_penalty = epoch_train_penalty / len(train_loader)
            
            history['train_loss'].append(avg_epoch_train_loss)
            history['train_base_loss'].append(avg_train_base_loss)
            history['train_penalty'].append(avg_train_penalty)
            
            # Validation step
            avg_epoch_val_loss = None
            avg_val_base_loss = None  
            avg_val_penalty = None
            if val_loader is not None:
                self.eval()  # Set model to evaluation mode
                epoch_val_loss = 0.0
                epoch_val_base_loss = 0.0
                epoch_val_penalty = 0.0
                
                # Don't use torch.no_grad() because we need gradients for forward pass derivatives
                # But disable gradient accumulation for parameters to prevent memory buildup
                for param in self.parameters():
                    param.grad = None
                
                for val_xi, val_yi in val_loader:
                    val_xi, val_yi = val_xi.to(self.device, self.dtype), val_yi.to(self.device, self.dtype)
                    
                    # Forward pass needs gradients for derivative computation
                    val_pred = self(val_xi)
                    
                    # Use same unified loss computation as training
                    total_loss, loss_components = self.compute_loss(val_pred, val_yi, val_xi, lambda_penalty)
                    
                    epoch_val_loss += loss_components['total_loss']
                    epoch_val_base_loss += loss_components['base_loss']
                    epoch_val_penalty += loss_components['penalty']
                    
                    # Clear any gradients that might have accumulated
                    for param in self.parameters():
                        param.grad = None
                
                avg_epoch_val_loss = epoch_val_loss / len(val_loader)
                avg_val_base_loss = epoch_val_base_loss / len(val_loader)
                avg_val_penalty = epoch_val_penalty / len(val_loader)
                
                history['val_loss'].append(avg_epoch_val_loss)
                history['val_base_loss'].append(avg_val_base_loss)
                history['val_penalty'].append(avg_val_penalty)
            
            # Log LR used for the epoch just completed (before scheduler.step())
            lr_for_epoch = optimizer.param_groups[0]['lr']
            history['lr'].append(lr_for_epoch)

            # Use validation loss for best model selection if available, otherwise training loss
            current_loss = avg_epoch_val_loss if avg_epoch_val_loss is not None else avg_epoch_train_loss
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                
                # Save model to temporary checkpoint directory
                save_model(self, temp_checkpoint_dir, overwrite=True)
                
                # Save history up to this point
                save_history(history, temp_checkpoint_dir, overwrite=True)
            
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_epoch_train_loss)
                else:
                    scheduler.step() 

            # Print progress with detailed loss breakdown
            if avg_epoch_val_loss is not None:
                print(f'Epoch {epoch}/{num_epochs_to_run}, Train Loss: {avg_epoch_train_loss:.6f} (Base: {avg_train_base_loss:.6f}, Penalty: {avg_train_penalty:.6f}), Val Loss: {avg_epoch_val_loss:.6f} (Base: {avg_val_base_loss:.6f}, Penalty: {avg_val_penalty:.6f}), LR: {lr_for_epoch:.2e}')
            else:
                print(f'Epoch {epoch}/{num_epochs_to_run}, Train Loss: {avg_epoch_train_loss:.6f} (Base: {avg_train_base_loss:.6f}, Penalty: {avg_train_penalty:.6f}), LR: {lr_for_epoch:.2e}')
            
        print("Training for this configuration finished.")
        
        # Load the best model from temporary checkpoint
        model_path = os.path.join(temp_checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            try:
                # Use load_model function to load the best checkpoint
                best_model = load_model(model_path, self.device, self.dtype, initialized_model=self)
                self.load_state_dict(best_model.state_dict())
                print(f"Successfully loaded best model from checkpoint at epoch {best_epoch}")
                
                # Clean up temporary checkpoint directory
                if os.path.exists(temp_checkpoint_dir):
                    shutil.rmtree(temp_checkpoint_dir)
                    print(f"Cleaned up temporary checkpoint directory: {temp_checkpoint_dir}")
                    
            except Exception as e:
                print(f"Failed to load best model from checkpoint: {e}")
                print("Continuing with current model state")
        else:
            print("No checkpoint found, continuing with current model state")
        
        return history

    def t_forward(self, t, x):
        return self.forward(x)
    
    def plot_lagrangian(self, x):
        return self.lagrangian(torch.Tensor(x)).cpu().detach()
    

def loss(pred, targ):
    """
    Legacy loss function - kept for backward compatibility.
    Consider using LNN.compute_loss() for new code as it includes regularization.
    """
    return torch.mean(torch.abs(pred - targ))

def custom_initialize_weights(model_instance):
    """
    Applies custom weight initialization to the LNN model instance.
    Specifically, initializes the last linear layer of the 'mlp' part
    of the model's 'network' attribute.
    """
    print("Applying custom weight initialization...")
    with torch.no_grad():
        last_linear_layer_mlp = None
        
        # Handle nn.DataParallel if model is wrapped
        network_to_inspect = model_instance.module.network if isinstance(model_instance, torch.nn.DataParallel) else model_instance.network
        
        # Initialize last layer of the main MLP (self.mlp in CombinedNetwork)
        if hasattr(network_to_inspect, 'mlp') and isinstance(network_to_inspect.mlp, torch.nn.Sequential):
            for layer in reversed(network_to_inspect.mlp):
                if isinstance(layer, torch.nn.Linear):
                    last_linear_layer_mlp = layer
                    break
            
            if last_linear_layer_mlp is not None:
                print("Initializing last layer of main MLP.")
                last_linear_layer_mlp.weight.data.uniform_(-0.01, 0.01)
                if last_linear_layer_mlp.bias is not None: # Your mlp's last layer has bias=False
                    last_linear_layer_mlp.bias.data.fill_(0.0)
            else:
                print("Warning: Could not find last linear layer in 'model.network.mlp' for custom initialization.")
        else:
            print("Warning: 'model.network.mlp' not found or not a Sequential module.")

        # Note: Your QuadraticMLP (self.qmlp in CombinedNetwork) uses default PyTorch initializations
        # for its layers, including its own last linear layer. If you also want to
        # initialize it similarly, you'd add another block here to inspect 'model.network.qmlp'.
        # For now, we are only re-applying the initialization you explicitly had for the main 'mlp'.