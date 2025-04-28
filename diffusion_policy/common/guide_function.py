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
            "l2_attractive_gradient": RepulsiveGradients.l2_attractive_gradient,
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
    def inverse_square_gradient(naction, obst_to_ee, scaling_factor=0.001, **kwargs):
        """
        Classic inverse square repulsion - stronger at close range, gradually decreases.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obst_to_ee (torch.Tensor): obst_to_ee positions (obst_to_ee_horizon, action_dim)
            
        Returns:
            torch.Tensor: Gradient to push away from obst_to_ee
        """
        if obst_to_ee.ndim == 2:
            assert naction.shape[2] == obst_to_ee.shape[1], "Action and obst_to_ee dimension mismatch"
            # Sample obst_to_ee positions to match prediction horizon
            indices = torch.linspace(0, obst_to_ee.shape[0]-1, naction.shape[1], dtype=int)
            obst_to_ee = torch.unsqueeze(obst_to_ee[indices], dim=0)  # (1, pred_horizon, action_dim)
        else:
            assert naction.shape[2] == obst_to_ee.shape[2], "Action and obst_to_ee dimension mismatch"
            indices = torch.linspace(0, obst_to_ee.shape[1]-1, naction.shape[1], dtype=int)
        # print("naction[..., 2]: ", naction[..., 2])
        # naction = naction/10
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            # Calculate distance between action and obst_to_ee (position only)
            position_diff = naction[:, :, :3] + obst_to_ee[:, :, :3]
            dist = torch.linalg.norm(position_diff, dim=2)  # (B, pred_horizon)
            safe_dist = torch.clamp(dist, min=1e-6)  # Prevent division by zero
            # # print("naction   ", naction[..., 2])
            # if (obst_to_ee[..., 2] < 0).any():
            #     print("obst_to_ee[:, 2] < 0, ")

            # if (position_diff[..., 2] < 0).any():
            #     print("position_diff[:, 2] < 0, ")
            
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
            
            # Apply condition: if |y_action - obst_to_ee[1]| > 0.1, set grad to 0
            # condition = torch.abs(obst_to_ee[:, :, 1]) > 0.05

            # valid_mask = torch.abs(obst_to_ee[:, :, 1]) < 0.1
            # if torch.any(valid_mask):
            #     obst_to_ee_z = obst_to_ee[:, :, 2][valid_mask]
            #     grad_z = grad[:, :, 2][valid_mask]

            #     # print(f"obst_to_ee_z values (|y|<0.1): {obst_to_ee_z}")
            #     # print(f"grad_z values (|y|<0.1): {grad_z}")

            #     # Check if their signs match
            #     diff_sign = (torch.sign(obst_to_ee_z) != torch.sign(grad_z))
            #     # if torch.any(diff_sign):
            #     #     print("Warn")


            # if torch.any(condition):
            #     # Check for obst_to_ee[:, :, 2] where condition is True
            #     grad[condition] = 0
            # if (grad[..., 2] > 0).any():
            #     print("grad[:, 2]: ")


        return -scaling_factor * grad  # Negate to push away from obst_to_ee
    
    @staticmethod
    def l2_attractive_gradient(naction, obj_to_ee, scaling_factor=0.001, **kwargs):
        """
        L2 norm attractive gradient - pulls the end effector toward the object.
        
        Args:
            naction (torch.Tensor): End effector trajectory (B, pred_horizon, action_dim)
            obj_to_ee (torch.Tensor): Object to end effector relative positions (obj_horizon, action_dim)
                                    or (B, obj_horizon, action_dim)
            scaling_factor (float): Factor to scale the gradient strength
            
        Returns:
            torch.Tensor: Gradient to pull the trajectory toward the object
        """
        if obj_to_ee.ndim == 2:
            assert naction.shape[2] == obj_to_ee.shape[1], "Action and obj_to_ee dimension mismatch"
            # Sample object positions to match prediction horizon
            indices = torch.linspace(0, obj_to_ee.shape[0]-1, naction.shape[1], dtype=int)
            obj_to_ee = torch.unsqueeze(obj_to_ee[indices], dim=0)  # (1, pred_horizon, action_dim)
        else:
            assert naction.shape[2] == obj_to_ee.shape[2], "Action and obj_to_ee dimension mismatch"
            indices = torch.linspace(0, obj_to_ee.shape[1]-1, naction.shape[1], dtype=int)
        
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            
            # Calculate distance between action and object (position only)
            position_diff = naction[:, :, :3] + obj_to_ee[:, :, :3]
            dist = torch.linalg.norm(position_diff, dim=2)  # (B, pred_horizon)
            
            # L2 norm attractive potential (directly proportional to distance)
            attractive_potential = dist
            total_potential = attractive_potential.sum(dim=1)
            
            # Calculate gradient
            grad = torch.autograd.grad(
                total_potential, 
                naction, 
                grad_outputs=torch.ones_like(total_potential),
                create_graph=False
            )[0]
            
            # Only apply gradient to position components
            grad[:, :, 3:] = 0
        print("grad: ", grad)
        # For attractive forces, we don't negate the gradient since we want to
        # move in the direction that reduces the distance (minimizes the L2 norm)
        return scaling_factor * grad


    @staticmethod
    def exponential_gradient(naction, obstacle, alpha=10.0, scaling_factor = 0.001, **kwargs):
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
        return -scaling_factor*grad  # Negate to push away from obstacle
    
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