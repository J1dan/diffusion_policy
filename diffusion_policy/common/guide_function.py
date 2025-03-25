import torch

class RepulsiveGradients:
    """
    A collection of repulsive gradient functions for obstacle avoidance.
    
    Each method calculates a gradient that pushes an end effector away from obstacles,
    with different characteristics for the repulsive field.
    """
    
    @staticmethod
    def compute_repulsive_gradient(naction, obstacle, method="inverse_square", **kwargs):
        """
        Compute repulsive gradient using the specified method.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obstacle (torch.Tensor): Obstacle positions (obstacle_horizon, action_dim)
            method (str): Name of the repulsion method to use
            **kwargs: Additional parameters specific to the chosen method
            
        Returns:
            torch.Tensor: Gradient to push the trajectory away from obstacles
        """
        methods = {
            "inverse_square": RepulsiveGradients.inverse_square_gradient,
            "exponential": RepulsiveGradients.exponential_gradient,
            "power": RepulsiveGradients.power_gradient,
            "gaussian": RepulsiveGradients.gaussian_gradient,
            "yukawa": RepulsiveGradients.yukawa_gradient,
            "distance_weighted": RepulsiveGradients.distance_weighted_gradient
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(methods.keys())}")
        
        return methods[method](naction, obstacle, **kwargs)
    
    @staticmethod
    def inverse_square_gradient(naction, obstacle, scaling_factor = 0.001, **kwargs):
        """
        Classic inverse square repulsion - stronger at close range, gradually decreases.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obstacle (torch.Tensor): Obstacle positions (obstacle_horizon, action_dim)
            
        Returns:
            torch.Tensor: Gradient to push away from obstacle
        """
        assert naction.shape[2] == obstacle.shape[1], "Action and obstacle dimension mismatch"
        
        # Sample obstacle positions to match prediction horizon
        indices = torch.linspace(0, obstacle.shape[0]-1, naction.shape[1], dtype=int)
        obstacle = torch.unsqueeze(obstacle[indices], dim=0)  # (1, pred_horizon, action_dim)
        
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            
            # Calculate distance between action and obstacle (position only)
            position_diff = naction[:, :, :3] - obstacle[:, :, :3]
            dist = torch.linalg.norm(position_diff, dim=2)  # (B, pred_horizon)
            safe_dist = torch.clamp(dist, min=1e-6)  # Prevent division by zero
            
            # Inverse square repulsive potential (1/r²)
            repulsive_potential = 1.0 / (safe_dist ** 2)
            total_potential = repulsive_potential.sum(dim=1)
            
            # Calculate gradient
            grad = torch.autograd.grad(
                total_potential, 
                naction, 
                grad_outputs=torch.ones_like(total_potential),
                create_graph=False
            )[0]

            # Only apply gradient to position components
            grad[:, :, 3:] = 0
            for axis in range(3):  # x, y, z axes
                # Get signs of position difference and gradient for this axis
                pos_diff_sign = torch.sign(position_diff[:, :, axis])
                grad_sign = torch.sign(-grad[:, :, axis])
                
                # Find misaligned points
                misaligned = (pos_diff_sign * grad_sign) < 0
                
                if torch.any(misaligned):
                    # Optional: print debug info about misalignments
                    misaligned_indices = torch.nonzero(misaligned)
                    print(f"Misaligned directions detected on axis {axis}")
                    print(f"Number of misalignments: {misaligned.sum().item()}")
            # Apply condition: if |y_action - obstacle[1]| > 0.1, set grad to 0
            # y_action = naction[:, :, 1]  # Extract y-component of action
            # condition = torch.abs(y_action - obstacle[:, :, 1]) > 0.1
            # grad[condition] = 0
        return -scaling_factor*grad  # Negate to push away from obstacle #-0.005 does not work well
    
    @staticmethod
    def exponential_gradient(naction, obstacle, alpha=10.0, **kwargs):
        """
        Exponential repulsion - very strong at close range with rapid falloff.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obstacle (torch.Tensor): Obstacle positions (obstacle_horizon, action_dim)
            alpha (float): Controls how rapidly repulsion decreases with distance
            
        Returns:
            torch.Tensor: Gradient to push away from obstacle
        """
        assert naction.shape[2] == obstacle.shape[1], "Action and obstacle dimension mismatch"
        
        # Sample obstacle positions to match prediction horizon
        indices = torch.linspace(0, obstacle.shape[0]-1, naction.shape[1], dtype=int)
        obstacle = torch.unsqueeze(obstacle[indices], dim=0)  # (1, pred_horizon, action_dim)
        
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            
            # Calculate distance between action and obstacle (position only)
            position_diff = naction[:, :, :3] - obstacle[:, :, :3]
            dist = torch.linalg.norm(position_diff, dim=2)  # (B, pred_horizon)
            
            # Exponential repulsive potential
            repulsive_potential = torch.exp(-(alpha * dist)**2)
            total_potential = repulsive_potential.sum(dim=1)
            
            # Calculate gradient
            grad = torch.autograd.grad(
                total_potential, 
                naction, 
                grad_outputs=torch.ones_like(total_potential),
                create_graph=False
            )[0]
            
            # Only apply gradient to position components
            grad[:, :, 3:] = 0



            for axis in range(3):  # x, y, z axes
                # Get signs of position difference and gradient for this axis
                pos_diff_sign = torch.sign(position_diff[:, :, axis])
                grad_sign = torch.sign(-grad[:, :, axis])
                
                # Find misaligned points
                misaligned = (pos_diff_sign * grad_sign) < 0
                
                if torch.any(misaligned):
                    # Optional: print debug info about misalignments
                    misaligned_indices = torch.nonzero(misaligned)
                    print(f"Misaligned directions detected on axis {axis}")
                    print(f"Number of misalignments: {misaligned.sum().item()}")
            # position_diff_unit = position_diff
            # grad_unit = -grad[..., :3]  # Negate the gradient

            # # Check if they have the same sign along each axis (x, y, z)
            # direction_check = torch.all(torch.sign(position_diff_unit) == torch.sign(grad_unit))

            # if not direction_check:
            #     raise ValueError("Direction of position_diff does not match -grad[..., :3]")
            # grad[:, :, :2] = 0 # Ignore x and y components
            # Apply condition: if |y_action - obstacle[1]| > 0.1, set grad to 0
            # y_action = naction[:, :, 1]  # Extract y-component of action
            # condition = torch.abs(y_action - obstacle[:, :, 1]) > 0.1
            # grad[condition] = 0
        # print("Grad torch:", grad)
        return -0.8*grad  # Negate to push away from obstacle
    
    @staticmethod
    def power_gradient(naction, obstacle, power=4, **kwargs):
        """
        Higher-power inverse distance - allows for stronger close-range repulsion.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obstacle (torch.Tensor): Obstacle positions (obstacle_horizon, action_dim)
            power (float): Power for the inverse distance function, higher values give
                           stronger close-range repulsion
            
        Returns:
            torch.Tensor: Gradient to push away from obstacle
        """
        assert naction.shape[2] == obstacle.shape[1], "Action and obstacle dimension mismatch"
        
        # Sample obstacle positions to match prediction horizon
        indices = torch.linspace(0, obstacle.shape[0]-1, naction.shape[1], dtype=int)
        obstacle = torch.unsqueeze(obstacle[indices], dim=0)  # (1, pred_horizon, action_dim)
        
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            
            # Calculate distance between action and obstacle (position only)
            position_diff = naction[:, :, :3] - obstacle[:, :, :3]
            dist = torch.linalg.norm(position_diff, dim=2)  # (B, pred_horizon)
            safe_dist = torch.clamp(dist, min=1e-6)  # Prevent division by zero
            
            # Inverse power repulsive potential (1/r^power)
            repulsive_potential = 1.0 / (safe_dist ** power)
            total_potential = repulsive_potential.sum(dim=1)
            
            # Calculate gradient
            grad = torch.autograd.grad(
                total_potential, 
                naction, 
                grad_outputs=torch.ones_like(total_potential),
                create_graph=False
            )[0]
            
            # Only apply gradient to position components
            grad[:, :, 3:] = 0
            
        return -grad  # Negate to push away from obstacle
    
    @staticmethod
    def gaussian_gradient(naction, obstacle, sigma=0.3, amplitude=1.0, **kwargs):
        """
        Gaussian repulsion - smooth, differentiable at all points.
        Less aggressive than inverse distance methods at very close range.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obstacle (torch.Tensor): Obstacle positions (obstacle_horizon, action_dim)
            sigma (float): Controls the width of the Gaussian
            amplitude (float): Controls the strength of repulsion
            
        Returns:
            torch.Tensor: Gradient to push away from obstacle
        """
        assert naction.shape[2] == obstacle.shape[1], "Action and obstacle dimension mismatch"
        
        # Sample obstacle positions to match prediction horizon
        indices = torch.linspace(0, obstacle.shape[0]-1, naction.shape[1], dtype=int)
        obstacle = torch.unsqueeze(obstacle[indices], dim=0)  # (1, pred_horizon, action_dim)
        
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            
            # Calculate distance between action and obstacle (position only)
            position_diff = naction[:, :, :3] - obstacle[:, :, :3]
            dist_squared = torch.sum(position_diff ** 2, dim=2)  # Squared distance
            
            # Gaussian repulsive potential
            repulsive_potential = amplitude * torch.exp(-0.5 * dist_squared / (sigma ** 2))
            total_potential = repulsive_potential.sum(dim=1)
            
            # Calculate gradient
            grad = torch.autograd.grad(
                total_potential, 
                naction, 
                grad_outputs=torch.ones_like(total_potential),
                create_graph=False
            )[0]
            
            # Only apply gradient to position components
            grad[:, :, 3:] = 0
            
        return -grad  # Negate to push away from obstacle
    
    @staticmethod
    def yukawa_gradient(naction, obstacle, alpha=3.0, **kwargs):
        """
        Yukawa potential - combines exponential decay with inverse distance.
        Good blend of strong close repulsion with faster falloff than inverse square.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obstacle (torch.Tensor): Obstacle positions (obstacle_horizon, action_dim)
            alpha (float): Controls decay rate
            
        Returns:
            torch.Tensor: Gradient to push away from obstacle
        """
        assert naction.shape[2] == obstacle.shape[1], "Action and obstacle dimension mismatch"
        
        # Sample obstacle positions to match prediction horizon
        indices = torch.linspace(0, obstacle.shape[0]-1, naction.shape[1], dtype=int)
        obstacle = torch.unsqueeze(obstacle[indices], dim=0)  # (1, pred_horizon, action_dim)
        
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            
            # Calculate distance between action and obstacle (position only)
            position_diff = naction[:, :, :3] - obstacle[:, :, :3]
            dist = torch.linalg.norm(position_diff, dim=2)  # (B, pred_horizon)
            safe_dist = torch.clamp(dist, min=1e-6)  # Prevent division by zero
            
            # Yukawa repulsive potential (e^(-αr)/r)
            repulsive_potential = torch.exp(-alpha * safe_dist) / safe_dist
            total_potential = repulsive_potential.sum(dim=1)
            
            # Calculate gradient
            grad = torch.autograd.grad(
                total_potential, 
                naction, 
                grad_outputs=torch.ones_like(total_potential),
                create_graph=False
            )[0]
            
            # Only apply gradient to position components
            grad[:, :, 3:] = 0
            
        return -grad  # Negate to push away from obstacle
    
    @staticmethod
    def distance_weighted_gradient(naction, obstacle, d0=0.5, power=2.0, **kwargs):
        """
        Distance-weighted approach - more control over the falloff behavior.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obstacle (torch.Tensor): Obstacle positions (obstacle_horizon, action_dim)
            d0 (float): Reference distance
            power (float): Power for falloff
            
        Returns:
            torch.Tensor: Gradient to push away from obstacle
        """
        assert naction.shape[2] == obstacle.shape[1], "Action and obstacle dimension mismatch"
        
        # Sample obstacle positions to match prediction horizon
        indices = torch.linspace(0, obstacle.shape[0]-1, naction.shape[1], dtype=int)
        obstacle = torch.unsqueeze(obstacle[indices], dim=0)  # (1, pred_horizon, action_dim)
        
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            
            # Calculate distance between action and obstacle (position only)
            position_diff = naction[:, :, :3] - obstacle[:, :, :3]
            dist = torch.linalg.norm(position_diff, dim=2)  # (B, pred_horizon)
            safe_dist = torch.clamp(dist, min=1e-6)  # Prevent division by zero
            
            # Distance-weighted potential (d0/r)^power
            repulsive_potential = (d0 / safe_dist) ** power
            total_potential = repulsive_potential.sum(dim=1)
            
            # Calculate gradient
            grad = torch.autograd.grad(
                total_potential, 
                naction, 
                grad_outputs=torch.ones_like(total_potential),
                create_graph=False
            )[0]
            
            # Only apply gradient to position components
            grad[:, :, 3:] = 0
            
        return -grad  # Negate to push away from obstacle