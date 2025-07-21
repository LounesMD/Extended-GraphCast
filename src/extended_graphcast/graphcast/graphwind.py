import jax
import jax.numpy as jnp
import haiku as hk
import xarray
from typing import Tuple, Dict, Optional
from graphcast import losses
from graphcast.graphcast import GraphCast, ModelConfig, TaskConfig
from graphcast import predictor_base
import optax

class Wind100mPredictor(hk.Module):
    """Auxiliary network for predicting 100m wind speed from GraphCast internals."""
    
    def __init__(self, 
                 hidden_size: int = 1024,
                 num_layers: int = 6,
                 use_grid_features: bool = True,
                 use_mesh_features: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_grid_features = use_grid_features
        self.use_mesh_features = use_mesh_features
        
    def __call__(self, 
                 latent_grid_nodes: jnp.ndarray,
                 latent_mesh_nodes: jnp.ndarray,
                 updated_latent_mesh_nodes: jnp.ndarray,
                 graphcast_predictions: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            latent_grid_nodes: [num_grid_nodes, batch, latent_size] from grid2mesh
            latent_mesh_nodes: [num_mesh_nodes, batch, latent_size] initial mesh features
            updated_latent_mesh_nodes: [num_mesh_nodes, batch, latent_size] after mesh GNN
            graphcast_predictions: [num_grid_nodes, batch, num_outputs] final predictions
            
        Returns:
            wind_100m: [num_grid_nodes, batch, 1] predictions for 100m wind speed
        """
        # Extract 10m wind from predictions (assuming standard GraphCast output order)
        # You'll need to adjust these indices based on your actual output configuration
        u10_idx = 0  # Index for 10m u component
        v10_idx = 1  # Index for 10m v component
        
        u10 = graphcast_predictions[..., u10_idx:u10_idx+1]
        v10 = graphcast_predictions[..., v10_idx:v10_idx+1]
        wind10_speed = jnp.sqrt(u10**2 + v10**2)
        
        # Start with grid features
        features = [latent_grid_nodes]
        
        if self.use_mesh_features:
            # Interpolate mesh features back to grid
            # This is a simplified version - in practice you'd use the mesh2grid connectivity
            # For now, we'll just use the grid features
            mesh_evolution = updated_latent_mesh_nodes - latent_mesh_nodes
            # You would properly interpolate mesh_evolution to grid here
            # features.append(interpolated_mesh_evolution)
        
        # Add physical features
        features.extend([wind10_speed, u10, v10])
        
        # Concatenate all features
        combined_features = jnp.concatenate(features, axis=-1)
        
        # MLP to predict 100m wind
        x = combined_features
        for i in range(self.num_layers):
            x = hk.Linear(self.hidden_size)(x)
            if i < self.num_layers - 1:
                x = jax.nn.swish(x)
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        
        # Final projection to wind speed
        wind_100m = hk.Linear(1)(x)
        
        # Apply physical constraint: 100m wind should generally be stronger than 10m
        # due to reduced surface friction
        wind_ratio = jax.nn.sigmoid(wind_100m) * 0.5 + 1.0  # Ratio between 1.0 and 1.5
        wind_100m = wind10_speed * wind_ratio
        
        return wind_100m


class GraphCastWith100mWind(GraphCast):
    """Extended GraphCast that also predicts 100m wind speed."""
    
    def __init__(self, model_config: ModelConfig, task_config: TaskConfig):
        super().__init__(model_config, task_config)
        
        # Initialize the wind predictor network
        self._wind_100m_predictor = Wind100mPredictor(
            hidden_size=model_config.latent_size // 2,
            num_layers=3,
            use_grid_features=True,
            use_mesh_features=True,
            name="wind_100m_predictor"
        )
        
        # Flag to determine if we should output 100m wind
        self._predict_100m_wind = True
        
    def __call__(self,
                 inputs: xarray.Dataset,
                 targets_template: xarray.Dataset,
                 forcings: xarray.Dataset,
                 is_training: bool = False,
                 ) -> xarray.Dataset:
        """Forward pass with optional 100m wind prediction."""
        
        self._maybe_init(inputs)
        
        # Convert inputs to grid node features
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
        
        # Run the standard GraphCast forward pass, but capture intermediates
        # Grid to mesh
        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(grid_node_features)
        
        # Mesh processing
        updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)
        
        # Mesh to grid
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes)
        
        # Get standard predictions
        predictions = self._grid_node_outputs_to_prediction(output_grid_nodes, targets_template)
        
        if self._predict_100m_wind:
            # Predict 100m wind using auxiliary network
            wind_100m_nodes = self._wind_100m_predictor(
                latent_grid_nodes=latent_grid_nodes,
                latent_mesh_nodes=latent_mesh_nodes,
                updated_latent_mesh_nodes=updated_latent_mesh_nodes,
                graphcast_predictions=output_grid_nodes
            )
            
            # Add to predictions dataset
            predictions = self._add_wind_100m_to_predictions(
                predictions, wind_100m_nodes, targets_template)
        
        return predictions
    
    def _add_wind_100m_to_predictions(self,
                                      predictions: xarray.Dataset,
                                      wind_100m_nodes: jnp.ndarray,
                                      targets_template: xarray.Dataset) -> xarray.Dataset:
        """Add 100m wind speed to the predictions dataset."""
        
        # Reshape from nodes to grid
        assert self._grid_lat is not None and self._grid_lon is not None
        grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])
        wind_100m_grid = wind_100m_nodes.reshape(
            grid_shape + wind_100m_nodes.shape[1:])
        
        # Create xarray DataArray
        # Remove the last dimension if it's size 1
        if wind_100m_grid.shape[-1] == 1:
            wind_100m_grid = wind_100m_grid[..., 0]
        
        wind_100m_da = xarray.DataArray(
            data=wind_100m_grid.transpose(2, 0, 1),  # [batch, lat, lon]
            dims=["batch", "lat", "lon"],
            coords={
                "batch": predictions.batch,
                "lat": predictions.lat,
                "lon": predictions.lon,
            }
        )
        
        # Add time dimension to match predictions format
        wind_100m_da = wind_100m_da.expand_dims({"time": 1}, axis=1)
        
        # Add to predictions
        predictions["wind_speed_100m"] = wind_100m_da
        
        return predictions
    
    def loss_and_predictions(self,
                            inputs: xarray.Dataset,
                            targets: xarray.Dataset,
                            forcings: xarray.Dataset,
                            ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
        """Compute loss including 100m wind if available in targets."""
        
        # Forward pass
        predictions = self(
            inputs, targets_template=targets, forcings=forcings, is_training=True)
        
        # Compute standard loss
        standard_vars = [v for v in targets.data_vars if v != "wind_speed_100m"]
        standard_loss = losses.weighted_mse_per_level(
            predictions[standard_vars], 
            targets[standard_vars],
            per_variable_weights={
                "2m_temperature": 1.0,
                "10m_u_component_of_wind": 0.1,
                "10m_v_component_of_wind": 0.1,
                "mean_sea_level_pressure": 0.1,
                "total_precipitation_6hr": 0.1,
            })
        
        # Add 100m wind loss if present in targets
        if "wind_speed_100m" in targets.data_vars and "wind_speed_100m" in predictions.data_vars:
            wind_100m_loss = losses.weighted_mse_per_level(
                predictions[["wind_speed_100m"]], 
                targets[["wind_speed_100m"]],
                per_variable_weights={"wind_speed_100m": 0.5}  # Adjust weight as needed
            )
            
            # Combine losses
            total_loss = standard_loss + wind_100m_loss
        else:
            total_loss = standard_loss
        
        return total_loss, predictions


# Training utilities
def create_wind_only_optimizer(model_params, learning_rate=1e-4):
    """Create optimizer that only updates wind predictor parameters."""
    
    # Identify wind predictor parameters
    wind_params_mask = hk.data_structures.map(
        lambda module_name, name, value: "wind_100m_predictor" in module_name,
        model_params
    )
    
    # Create optimizer with masking
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate),
        optax.masked(wind_params_mask)
    )
    
    return optimizer


def train_wind_predictor_only(model: GraphCastWith100mWind,
                             params: Dict,
                             train_data_loader,
                             num_steps: int = 1000):
    """Train only the wind predictor while keeping GraphCast frozen."""
    
    # Create optimizer for wind predictor only
    optimizer = create_wind_only_optimizer(params)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, inputs, targets, forcings):
        def loss_fn(params):
            predictions = model.apply(params, inputs, targets, forcings, is_training=True)
            
            # Only compute loss on 100m wind
            if "wind_speed_100m" in predictions.data_vars and "wind_speed_100m" in targets.data_vars:
                loss = losses.weighted_mse_per_level(
                    predictions[["wind_speed_100m"]], 
                    targets[["wind_speed_100m"]],
                    per_variable_weights={"wind_speed_100m": 1.0}
                )
            else:
                loss = 0.0
            
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    # Training loop
    for step in range(num_steps):
        batch = next(train_data_loader)
        params, opt_state, loss = train_step(
            params, opt_state, 
            batch['inputs'], 
            batch['targets'], 
            batch['forcings']
        )
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
    
    return params