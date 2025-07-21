import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from graphcast import icosahedral_mesh
from graphcast import model_utils

def visualize_graphcast_meshes(model_config):
    """Visualize the icosahedral meshes used in GraphCast."""
    
    # Get the hierarchy of meshes
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=model_config.mesh_size)
    
    # Create figure with subplots for different mesh levels
    fig = plt.figure(figsize=(16, 8))
    
    # Plot the finest mesh on a map projection
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    finest_mesh = meshes[-1]  # Last mesh is the finest
    
    # Convert 3D cartesian coordinates to lat/lon
    vertices = finest_mesh.vertices
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        vertices[:, 0], vertices[:, 1], vertices[:, 2])
    mesh_lat, mesh_lon = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta)
    
    # Plot mesh nodes on map
    ax1.scatter(mesh_lon, mesh_lat, c='red', s=1, transform=ccrs.PlateCarree(), 
                alpha=0.5, label=f'{len(vertices)} mesh nodes')
    ax1.add_feature(cfeature.COASTLINE, alpha=0.5)
    ax1.add_feature(cfeature.BORDERS, alpha=0.3)
    ax1.set_global()
    ax1.set_title(f'Finest Mesh (Level {model_config.mesh_size}): {len(vertices)} nodes')
    ax1.legend()
    
    # Plot 3D view of multi-resolution meshes
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Plot each mesh level with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(meshes)))
    
    for i, mesh in enumerate(meshes):
        vertices = mesh.vertices * (0.9 + i * 0.02)  # Slightly offset each level
        
        # Plot vertices
        ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   c=[colors[i]], s=20/(i+1), alpha=0.7,
                   label=f'Level {i}: {len(vertices)} nodes')
        
        # Optionally plot edges for coarser meshes
        if i < 3:  # Only show edges for first few levels
            faces = mesh.faces
            for face in faces[::max(1, len(faces)//100)]:  # Sample faces for visibility
                triangle = vertices[face]
                triangle = np.vstack([triangle, triangle[0]])  # Close the triangle
                ax2.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                        c=colors[i], alpha=0.3, linewidth=0.5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Multi-Resolution Icosahedral Meshes')
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig, meshes


def plot_mesh_statistics(meshes):
    """Plot statistics about the mesh hierarchy."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Number of nodes and faces per level
    levels = list(range(len(meshes)))
    num_nodes = [len(mesh.vertices) for mesh in meshes]
    num_faces = [len(mesh.faces) for mesh in meshes]
    
    ax1.semilogy(levels, num_nodes, 'o-', label='Vertices', markersize=8)
    ax1.semilogy(levels, num_faces, 's-', label='Faces', markersize=8)
    ax1.set_xlabel('Refinement Level')
    ax1.set_ylabel('Count')
    ax1.set_title('Mesh Complexity by Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Edge lengths distribution for finest mesh
    finest_mesh = meshes[-1]
    senders, receivers = icosahedral_mesh.faces_to_edges(finest_mesh.faces)
    edge_lengths = np.linalg.norm(
        finest_mesh.vertices[senders] - finest_mesh.vertices[receivers], axis=1)
    
    ax2.hist(edge_lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(edge_lengths.mean(), color='red', linestyle='--', 
                label=f'Mean: {edge_lengths.mean():.3f}')
    ax2.axvline(edge_lengths.max(), color='orange', linestyle='--', 
                label=f'Max: {edge_lengths.max():.3f}')
    ax2.set_xlabel('Edge Length')
    ax2.set_ylabel('Count')
    ax2.set_title('Edge Length Distribution (Finest Mesh)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def visualize_graphcast_meshes_direct(meshes):
    """Visualize the icosahedral meshes directly from the meshes list."""
    
    # Create figure with subplots for different mesh levels
    fig = plt.figure(figsize=(16, 8))
    
    # Plot the finest mesh on a map projection
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    finest_mesh = meshes[-1]  # Last mesh is the finest
    
    # Convert 3D cartesian coordinates to lat/lon
    vertices = finest_mesh.vertices
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        vertices[:, 0], vertices[:, 1], vertices[:, 2])
    mesh_lat, mesh_lon = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta)
    
    # Plot mesh nodes on map
    ax1.scatter(mesh_lon, mesh_lat, c='red', s=1, transform=ccrs.PlateCarree(), 
                alpha=0.5, label=f'{len(vertices)} mesh nodes')
    ax1.add_feature(cfeature.COASTLINE, alpha=0.5)
    ax1.add_feature(cfeature.BORDERS, alpha=0.3)
    ax1.set_global()
    ax1.set_title(f'Finest Mesh: {len(vertices)} nodes')
    ax1.legend()
    
    # Plot 3D view
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Plot each mesh level with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(meshes)))
    
    for i, mesh in enumerate(meshes):
        vertices = mesh.vertices * (0.9 + i * 0.02)  # Slightly offset each level
        
        # Plot vertices
        ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   c=[colors[i]], s=20/(i+1), alpha=0.7,
                   label=f'Level {i}: {len(vertices)} nodes')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Multi-Resolution Icosahedral Meshes')
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig

def visualize_mesh_levels_separately(meshes):
    """Visualize each icosahedral mesh level in a separate subplot."""
    
    n_levels = len(meshes)
    fig = plt.figure(figsize=(20, 4 * n_levels))
    
    for level, mesh in enumerate(meshes):
        # Create two subplots for each level: 3D view and map projection
        ax_3d = fig.add_subplot(n_levels, 3, level * 3 + 1, projection='3d')
        ax_map = fig.add_subplot(n_levels, 3, level * 3 + 2, projection=ccrs.Robinson())
        ax_close = fig.add_subplot(n_levels, 3, level * 3 + 3, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Convert to lat/lon for map
        mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
            vertices[:, 0], vertices[:, 1], vertices[:, 2])
        mesh_lat, mesh_lon = model_utils.spherical_to_lat_lon(
            phi=mesh_phi, theta=mesh_theta)
        
        # 1. 3D view with edges
        ax_3d.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                     c='red', s=50, alpha=0.8)
        
        # Draw edges (sample them for higher levels to avoid clutter)
        max_edges_to_draw = 10000
        if len(faces) > max_edges_to_draw // 3:
            face_indices = np.random.choice(len(faces), max_edges_to_draw // 3, replace=False)
            sampled_faces = faces[face_indices]
        else:
            sampled_faces = faces
            
        for face in sampled_faces:
            triangle = vertices[face]
            triangle = np.vstack([triangle, triangle[0]])
            ax_3d.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                      'b-', alpha=0.3, linewidth=0.5)
        # for face in sampled_faces:
        #     face_lons = mesh_lon[face]
        #     face_lats = mesh_lat[face]

        #     # Close the triangle
        #     triangle_lons = np.append(face_lons, face_lons[0])
        #     triangle_lats = np.append(face_lats, face_lats[0])

        #     # Handle antimeridian wraparound
        #     for i in range(3):
        #         lon1, lon2 = triangle_lons[i], triangle_lons[i+1]
        #         lat1, lat2 = triangle_lats[i], triangle_lats[i+1]

        #         # Check for wraparound
        #         if abs(lon1 - lon2) > 180:
        #             if lon1 > lon2:
        #                 lon2 += 360
        #             else:
        #                 lon1 += 360

        #         ax_map.plot([lon1, lon2], [lat1, lat2], 'b-', linewidth=0.3,
        #                     alpha=0.4, transform=ccrs.PlateCarree())

        #         ax_3d.set_title(f'Level {level}: 3D Mesh\n{len(vertices)} nodes, {len(faces)} faces')
        #         ax_3d.set_xlabel('X')
        #         ax_3d.set_ylabel('Y')
        #         ax_3d.set_zlabel('Z')
        
        # 2. Map projection with edges
        ax_map.scatter(mesh_lon, mesh_lat, c='red', s=20/(level+1), 
                    transform=ccrs.PlateCarree(), alpha=0.7)

        # Draw edges (optionally sampled to reduce clutter)
        if len(faces) > max_edges_to_draw // 3:
            face_indices = np.random.choice(len(faces), max_edges_to_draw // 3, replace=False)
            sampled_faces = faces[face_indices]
        else:
            sampled_faces = faces

        for face in sampled_faces:
            face_lons = mesh_lon[face]
            face_lats = mesh_lat[face]

            # Close the triangle
            triangle_lons = np.append(face_lons, face_lons[0])
            triangle_lats = np.append(face_lats, face_lats[0])

            ax_map.plot(triangle_lons, triangle_lats, 'b-', linewidth=0.3,
                        alpha=0.4, transform=ccrs.PlateCarree())

        ax_map.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax_map.add_feature(cfeature.BORDERS, alpha=0.3)
        ax_map.set_global()
        ax_map.set_title(f'Level {level}: Geographic Distribution')
        
        # 3. Close-up view of mesh structure
        # Zoom in on a small region to show mesh detail
        ax_close.set_xlim([0.8, 1.0])
        ax_close.set_ylim([-0.1, 0.1])
        ax_close.set_zlim([-0.1, 0.1])
        
        # Find vertices in this region
        mask = (vertices[:, 0] > 0.8) & (vertices[:, 0] < 1.0) & \
               (np.abs(vertices[:, 1]) < 0.1) & (np.abs(vertices[:, 2]) < 0.1)
        
        if np.any(mask):
            local_vertices = vertices[mask]
            ax_close.scatter(local_vertices[:, 0], local_vertices[:, 1], 
                           local_vertices[:, 2], c='red', s=100, alpha=1.0)
            
            # Draw edges in this region
            for face in faces:
                if np.any(mask[face]):
                    triangle = vertices[face]
                    triangle = np.vstack([triangle, triangle[0]])
                    ax_close.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                                'b-', alpha=0.5, linewidth=1.0)
        
        ax_close.set_title(f'Level {level}: Mesh Detail (Zoomed)')
        ax_close.set_xlabel('X')
        ax_close.set_ylabel('Y')
        ax_close.set_zlabel('Z')
        
        # Add text info
        info_text = f"Refinement factor from previous: ~4x" if level > 0 else "Base icosahedron"
        fig.text(0.95, 0.85 - level * (1.0/n_levels), info_text, 
                fontsize=10, ha='right', transform=fig.transFigure)
    
    plt.tight_layout()
    return fig


def visualize_mesh_hierarchy_connections(meshes):
    """Visualize how vertices from coarser meshes relate to finer meshes."""
    
    fig = plt.figure(figsize=(15, 10))
    
    # Show the refinement pattern
    ax = fig.add_subplot(111, projection='3d')
    
    # Take a single face from the coarsest mesh and show its refinement
    coarse_mesh = meshes[0]
    if len(meshes) > 1:
        fine_mesh = meshes[1]
        
        # Pick one face from coarse mesh
        face_idx = 0
        coarse_face = coarse_mesh.faces[face_idx]
        coarse_triangle = coarse_mesh.vertices[coarse_face]
        
        # Plot coarse triangle
        triangle = np.vstack([coarse_triangle, coarse_triangle[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                'r-', linewidth=3, label='Coarse mesh face')
        ax.scatter(coarse_triangle[:, 0], coarse_triangle[:, 1], 
                  coarse_triangle[:, 2], c='red', s=200, zorder=5)
        
        # Find and plot fine mesh vertices within this triangle
        # (This is approximate - actual implementation would need proper containment test)
        center = coarse_triangle.mean(axis=0)
        distances = np.linalg.norm(fine_mesh.vertices - center[None, :], axis=1)
        nearby_mask = distances < 0.3  # Approximate
        
        nearby_vertices = fine_mesh.vertices[nearby_mask]
        ax.scatter(nearby_vertices[:, 0], nearby_vertices[:, 1], 
                  nearby_vertices[:, 2], c='blue', s=50, alpha=0.7,
                  label='Fine mesh vertices')
        
        # Draw some fine mesh edges
        for face in fine_mesh.faces:
            if np.any(nearby_mask[face]):
                fine_triangle = fine_mesh.vertices[face]
                fine_triangle = np.vstack([fine_triangle, fine_triangle[0]])
                ax.plot(fine_triangle[:, 0], fine_triangle[:, 1], fine_triangle[:, 2], 
                       'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh Refinement Pattern: One Coarse Face → Multiple Fine Faces')
    ax.legend()
    
    return fig


def visualize_input_grid(grid_lat, grid_lon):
    """Visualize the input latitude-longitude grid."""
    
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Global view of grid points
    ax1 = fig.add_subplot(2, 3, 1, projection=ccrs.Robinson())
    
    # Create meshgrid for visualization
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Plot grid points (subsample for visibility)
    step = max(1, len(grid_lat) // 50)  # Show ~50 points in each direction
    ax1.scatter(lon_mesh[::step, ::step], lat_mesh[::step, ::step], 
                c='blue', s=1, transform=ccrs.PlateCarree(), alpha=0.5)
    
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, alpha=0.5)
    ax1.set_global()
    ax1.set_title(f'Input Grid: {len(grid_lat)} × {len(grid_lon)} = {len(grid_lat) * len(grid_lon):,} points\n'
                  f'(showing every {step} points)')
    
    # 2. 3D sphere view
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    
    # Convert to 3D cartesian (subsample)
    lon_rad = np.deg2rad(lon_mesh[::step, ::step].flatten())
    lat_rad = np.deg2rad(lat_mesh[::step, ::step].flatten())
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    ax2.scatter(x, y, z, c='blue', s=1, alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Grid Points on Sphere')
    
    # 3. Regional zoom (e.g., Europe)
    ax3 = fig.add_subplot(2, 3, 3, projection=ccrs.PlateCarree())
    ax3.set_extent([-10, 40, 35, 70], crs=ccrs.PlateCarree())
    
    # Plot all points in this region
    ax3.scatter(lon_mesh, lat_mesh, c='blue', s=1, 
                transform=ccrs.PlateCarree(), alpha=0.7)
    
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS)
    ax3.add_feature(cfeature.OCEAN, alpha=0.3)
    ax3.add_feature(cfeature.LAND, alpha=0.3)
    ax3.gridlines(draw_labels=True)
    ax3.set_title('Regional View: Europe\n(showing all grid points)')
    
    # 4. Polar region (showing convergence)
    ax4 = fig.add_subplot(2, 3, 4, projection=ccrs.NorthPolarStereo())
    ax4.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    
    ax4.scatter(lon_mesh, lat_mesh, c='blue', s=1, 
                transform=ccrs.PlateCarree(), alpha=0.5)
    
    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.LAND, alpha=0.3)
    ax4.gridlines()
    ax4.set_title('Polar View: Grid Convergence')
    
    # 5. Grid spacing analysis
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Calculate actual distances between grid points at different latitudes
    latitudes_to_check = [0, 30, 60, 80]
    colors = ['red', 'green', 'blue', 'purple']
    
    for lat, color in zip(latitudes_to_check, colors):
        # Convert to numpy array if it's xarray
        grid_lat_np = grid_lat.values if hasattr(grid_lat, 'values') else grid_lat
        lat_idx = np.argmin(np.abs(grid_lat_np - lat))
        actual_lat = grid_lat_np[lat_idx]
        
        # Distance between longitude points at this latitude
        grid_lon_np = grid_lon.values if hasattr(grid_lon, 'values') else grid_lon
        R = 6371  # Earth radius in km
        lon_spacing = np.diff(grid_lon_np)[0] if len(grid_lon_np) > 1 else 0
        distance_km = R * np.cos(np.deg2rad(actual_lat)) * np.deg2rad(lon_spacing)
        
        ax5.axhline(distance_km, color=color, linestyle='-', 
                    label=f'Lat {actual_lat:.1f}°: {distance_km:.1f} km')
    
    # Also show latitude spacing
    lat_spacing = np.diff(grid_lat_np)[0] if len(grid_lat_np) > 1 else 0
    lat_distance_km = R * np.deg2rad(lat_spacing)
    ax5.axhline(lat_distance_km, color='black', linestyle='--', 
                label=f'Lat spacing: {lat_distance_km:.1f} km')
    
    ax5.set_ylabel('Distance (km)')
    ax5.set_title('Grid Point Spacing by Latitude')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Grid vs Mesh density comparison
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Grid density (points per unit area on sphere)
    total_grid_points = len(grid_lat) * len(grid_lon)
    earth_surface_area = 4 * np.pi  # Unit sphere
    grid_density = total_grid_points / earth_surface_area
    
    ax6.text(0.1, 0.8, f'Grid Statistics:', fontsize=12, fontweight='bold')
    ax6.text(0.1, 0.7, f'Resolution: {len(grid_lat)} × {len(grid_lon)}', fontsize=10)
    ax6.text(0.1, 0.6, f'Total points: {total_grid_points:,}', fontsize=10)
    ax6.text(0.1, 0.5, f'Lat spacing: {lat_spacing:.2f}°', fontsize=10)
    ax6.text(0.1, 0.4, f'Lon spacing: {lon_spacing:.2f}°', fontsize=10)
    ax6.text(0.1, 0.3, f'Density: {grid_density:.1f} points/steradian', fontsize=10)
    
    ax6.text(0.1, 0.1, 'Note: Grid spacing varies with latitude\n(closer at poles)',
             fontsize=10, style='italic', color='red')
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_grid_mesh_comparison(grid_lat, grid_lon, meshes):
    """Compare the input grid with the mesh structure."""
    
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Grid points
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
    step = max(1, len(grid_lat) // 100)
    
    ax1.scatter(lon_mesh[::step, ::step], lat_mesh[::step, ::step], 
                c='blue', s=0.5, transform=ccrs.PlateCarree(), 
                alpha=0.5, label=f'Grid: {len(grid_lat) * len(grid_lon):,} points')
    
    # 2. Mesh nodes on same plot
    finest_mesh = meshes[-1]
    vertices = finest_mesh.vertices
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        vertices[:, 0], vertices[:, 1], vertices[:, 2])
    mesh_lat, mesh_lon = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta)
    
    ax1.scatter(mesh_lon, mesh_lat, c='red', s=0.5, 
                transform=ccrs.PlateCarree(), alpha=0.5,
                label=f'Mesh: {len(vertices):,} nodes')
    
    ax1.add_feature(cfeature.COASTLINE, alpha=0.5)
    ax1.set_global()
    ax1.set_title('Grid Points vs Mesh Nodes')
    ax1.legend()
    
    # 2. Density comparison
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Calculate densities at different latitudes
    lat_bands = np.arange(-90, 91, 10)
    grid_counts = []
    mesh_counts = []
    
    for i in range(len(lat_bands) - 1):
        lat_min, lat_max = lat_bands[i], lat_bands[i + 1]
        
        # Count grid points in this band
        grid_mask = (lat_mesh >= lat_min) & (lat_mesh < lat_max)
        grid_counts.append(np.sum(grid_mask))
        
        # Count mesh nodes in this band
        mesh_mask = (mesh_lat >= lat_min) & (mesh_lat < lat_max)
        mesh_counts.append(np.sum(mesh_mask))
    
    lat_centers = (lat_bands[:-1] + lat_bands[1:]) / 2
    
    ax2.plot(lat_centers, grid_counts, 'b-', label='Grid points', linewidth=2)
    ax2.plot(lat_centers, mesh_counts, 'r-', label='Mesh nodes', linewidth=2)
    
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Number of points/nodes')
    ax2.set_title('Point Density by Latitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text showing key differences
    ax2.text(0.02, 0.98, 
             f'Grid: Higher density at poles\n'
             f'Mesh: ~Uniform density everywhere',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def plot_mesh_density_map(mesh, center_lat, center_lon, level):
    """Create a heatmap showing vertex density."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Convert vertices to lat/lon
    vertices = mesh.vertices
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        vertices[:, 0], vertices[:, 1], vertices[:, 2])
    mesh_lat, mesh_lon = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta)
    
    # Create density heatmap
    extent = [center_lon - 60, center_lon + 60, 
              center_lat - 40, center_lat + 40]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Hexbin for density
    antipodal_idx = len(vertices) - 1
    regional_mask = np.arange(len(vertices)) != antipodal_idx
    
    if np.sum(regional_mask) > 10:
        hb = ax.hexbin(
            mesh_lon[regional_mask], 
            mesh_lat[regional_mask],
            gridsize=30, 
            cmap='YlOrRd',
            transform=ccrs.PlateCarree(),
            mincnt=1
        )
        cb = plt.colorbar(hb, ax=ax, label='Vertices per hex cell')
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    ax.plot(center_lon, center_lat, 'b*', markersize=20, 
            transform=ccrs.PlateCarree(), label='Center')
    
    ax.set_title(f'Level {level}: Vertex Density Heatmap\n'
                 f'Total vertices: {len(vertices)} (excluding antipodal)')
    ax.legend()
    
    return fig

def visualize_adaptive_strategies():
    """Compare different mesh strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    strategies = [
        ("Original Icosahedral", "Even distribution"),
        ("Radial Rings", "Natural focus on center"),
        ("Information Spokes", "Direct paths to center"),
        ("Adaptive Refinement", "Distance-based subdivision")
    ]
    
    for ax, (title, desc) in zip(axes.flat, strategies):
        ax.set_title(f"{title}\n{desc}", fontsize=12)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        # Draw example mesh pattern
        # (Implementation depends on strategy)
    
    plt.tight_layout()
    return fig


# def visualize_regional_mesh_details(meshes, center_lat = 48.0, center_lon =  -2.75, zoom_radius_deg=40):
#     """Visualize each mesh level with focus on the regional area."""

#     n_levels = len(meshes)
#     fig = plt.figure(figsize=(20, 4 * n_levels))
    
#     for level, mesh in enumerate(meshes):
#         # Create subplots: global view, regional zoom, and ultra-zoom
#         ax_global = fig.add_subplot(n_levels, 4, level * 4 + 1, projection=ccrs.Robinson())
#         ax_regional = fig.add_subplot(n_levels, 4, level * 4 + 2, projection=ccrs.PlateCarree())
#         ax_close = fig.add_subplot(n_levels, 4, level * 4 + 3, projection=ccrs.PlateCarree())
#         ax_3d_regional = fig.add_subplot(n_levels, 4, level * 4 + 4, projection='3d')
#         vertices = mesh.vertices
#         faces = mesh.faces
        
#         # Convert to lat/lon
#         mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
#             vertices[:, 0], vertices[:, 1], vertices[:, 2])
#         mesh_lat, mesh_lon = model_utils.spherical_to_lat_lon(
#             phi=mesh_phi, theta=mesh_theta)
        
#         # Identify regional vertices (exclude antipodal point)
#         antipodal_idx = len(vertices) - 1
#         is_regional = np.arange(len(vertices)) != antipodal_idx
        
#         # 1. Global view (for context)
#         ax_global.scatter(mesh_lon[is_regional], mesh_lat[is_regional], 
#                          c='red', s=5, transform=ccrs.PlateCarree(), alpha=0.7)
#         ax_global.scatter(mesh_lon[antipodal_idx], mesh_lat[antipodal_idx], 
#                          c='blue', s=20, transform=ccrs.PlateCarree())
#         ax_global.add_feature(cfeature.COASTLINE, alpha=0.5)
#         ax_global.set_global()
#         ax_global.set_title(f'Level {level}: Global View\n{len(vertices)} vertices')
        
#         # 2. Regional view
#         regional_extent = [
#             center_lon - zoom_radius_deg, 
#             center_lon + zoom_radius_deg,
#             center_lat - zoom_radius_deg * 0.7,  
#             center_lat + zoom_radius_deg * 0.7
#         ]
#         ax_regional.set_extent(regional_extent, crs=ccrs.PlateCarree())
        
#         # Find vertices in regional view
#         regional_mask = (
#             (mesh_lon >= regional_extent[0]) & 
#             (mesh_lon <= regional_extent[1]) &
#             (mesh_lat >= regional_extent[2]) & 
#             (mesh_lat <= regional_extent[3]) &
#             is_regional
#         )
#         # regional_mask = None
#         # Plot vertices
#         if np.any(regional_mask):
#             ax_regional.scatter(mesh_lon[regional_mask], mesh_lat[regional_mask], 
#                               c='red', s=30, transform=ccrs.PlateCarree(), alpha=0.8)
        
#         # Draw edges in regional view
#         edges_drawn = 0
#         max_edges = 500  # Limit edges for clarity
        
#         for face in faces:
#             # if antipodal_idx not in face: #  and edges_drawn < max_edges:
#             face_lons = mesh_lon[face]
#             face_lats = mesh_lat[face]
            
#             # Check if face is in regional view
#             if (np.any((face_lons >= regional_extent[0]) & 
#                         (face_lons <= regional_extent[1]) &
#                         (face_lats >= regional_extent[2]) & 
#                         (face_lats <= regional_extent[3]))):
                
#                 # Close the triangle
#                 triangle_lons = np.append(face_lons, face_lons[0])
#                 triangle_lats = np.append(face_lats, face_lats[0])
                
#                 ax_regional.plot(triangle_lons, triangle_lats, 'b-', 
#                                 alpha=0.3, linewidth=0.5, transform=ccrs.PlateCarree())
#                 edges_drawn += 1
        
#         ax_regional.add_feature(cfeature.COASTLINE)
#         ax_regional.add_feature(cfeature.BORDERS)
#         ax_regional.add_feature(cfeature.OCEAN, alpha=0.3)
#         ax_regional.add_feature(cfeature.LAND, alpha=0.3)
#         ax_regional.gridlines(draw_labels=True, alpha=0.3)
#         ax_regional.set_title(f'Level {level}: Regional Zoom')
        
#         # Mark center
#         ax_regional.plot(center_lon, center_lat, 'g*', markersize=15, 
#                         transform=ccrs.PlateCarree())
        
#         # 3. Ultra close-up (10 degree box)
#         close_extent = [
#             center_lon - 10, 
#             center_lon + 10,
#             center_lat - 10,  
#             center_lat + 10
#         ]
#         ax_close.set_extent(close_extent, crs=ccrs.PlateCarree())
        
#         # Find vertices in close view
#         close_mask = (
#             (mesh_lon >= close_extent[0]) & 
#             (mesh_lon <= close_extent[1]) &
#             (mesh_lat >= close_extent[2]) & 
#             (mesh_lat <= close_extent[3]) &
#             is_regional
#         )
        
#         if np.any(close_mask):
#             ax_close.scatter(mesh_lon[close_mask], mesh_lat[close_mask], 
#                            c='red', s=50, transform=ccrs.PlateCarree())
            
#             # Draw ALL edges in this small region
#             for face in faces:
#                 if antipodal_idx not in face:
#                     face_lons = mesh_lon[face]
#                     face_lats = mesh_lat[face]
                    
#                     if np.all((face_lons >= close_extent[0] - 5) & 
#                              (face_lons <= close_extent[1] + 5) &
#                              (face_lats >= close_extent[2] - 5) & 
#                              (face_lats <= close_extent[3] + 5)):
                        
#                         triangle_lons = np.append(face_lons, face_lons[0])
#                         triangle_lats = np.append(face_lats, face_lats[0])
                        
#                         ax_close.plot(triangle_lons, triangle_lats, 'b-', 
#                                     alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
        
#         ax_close.add_feature(cfeature.COASTLINE, linewidth=2)
#         ax_close.add_feature(cfeature.BORDERS, linewidth=1)
#         ax_close.add_feature(cfeature.OCEAN, alpha=0.3)
#         ax_close.add_feature(cfeature.LAND, alpha=0.3)
#         ax_close.gridlines(draw_labels=True)
#         ax_close.set_title(f'Level {level}: Close-up Detail')
#         ax_close.plot(center_lon, center_lat, 'g*', markersize=20, 
#                      transform=ccrs.PlateCarree())
        
#         # 4. 3D view of regional area only
#         # Convert regional lat/lon box to 3D
#         regional_3d_mask = regional_mask
        
#         if np.any(regional_3d_mask):
#             regional_vertices = vertices[regional_3d_mask]
            
#             # Center and zoom the 3D view
#             center_3d = np.mean(regional_vertices, axis=0)
#             regional_vertices_centered = regional_vertices - center_3d
            
#             ax_3d_regional.scatter(
#                 regional_vertices_centered[:, 0], 
#                 regional_vertices_centered[:, 1], 
#                 regional_vertices_centered[:, 2], 
#                 c='red', s=30, alpha=0.8
#             )
            
#             # Draw some edges
#             edges_drawn_3d = 0
#             for face in faces:
#                 if antipodal_idx not in face and np.all(regional_3d_mask[face]) and edges_drawn_3d < 200:
#                     triangle = vertices[face] - center_3d
#                     triangle = np.vstack([triangle, triangle[0]])
#                     ax_3d_regional.plot(
#                         triangle[:, 0], triangle[:, 1], triangle[:, 2], 
#                         'b-', alpha=0.3, linewidth=0.5
#                     )
#                     edges_drawn_3d += 1
        
#         ax_3d_regional.set_xlabel('X')
#         ax_3d_regional.set_ylabel('Y')
#         ax_3d_regional.set_zlabel('Z')
#         ax_3d_regional.set_title(f'Level {level}: 3D Regional')
        
#         # Info text
#         vertices_in_region = np.sum(close_mask)
#         info_text = f"Vertices in ±10° box: {vertices_in_region}"
#         fig.text(0.02, 0.90 - level * (1.0/n_levels), info_text, 
#                 fontsize=12, transform=fig.transFigure)
    
#     plt.tight_layout()
#     return fig


def visualize_regional_mesh_details(meshes, center_lat=48.0, center_lon=-2.75, zoom_radius_deg=40):
    """Visualize each mesh level with focus on the regional area."""

    n_levels = len(meshes)
    fig = plt.figure(figsize=(20, 4 * n_levels))

    for level, mesh in enumerate(meshes):
        ax_global = fig.add_subplot(n_levels, 4, level * 4 + 1, projection=ccrs.Robinson())
        ax_regional = fig.add_subplot(n_levels, 4, level * 4 + 2, projection=ccrs.PlateCarree())
        ax_close = fig.add_subplot(n_levels, 4, level * 4 + 3, projection=ccrs.PlateCarree())
        ax_3d_regional = fig.add_subplot(n_levels, 4, level * 4 + 4, projection='3d')

        vertices = mesh.vertices
        faces = mesh.faces

        # Convert to lat/lon
        mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
            vertices[:, 0], vertices[:, 1], vertices[:, 2])
        mesh_lat, mesh_lon = model_utils.spherical_to_lat_lon(phi=mesh_phi, theta=mesh_theta)

        # Identify regional vertices (exclude antipodal point)
        antipodal_idx = len(vertices) - 1
        is_regional = np.arange(len(vertices)) != antipodal_idx

        # 1. Global view
        ax_global.scatter(mesh_lon[is_regional], mesh_lat[is_regional],
                          c='red', s=5, transform=ccrs.PlateCarree(), alpha=0.7)
        ax_global.scatter(mesh_lon[antipodal_idx], mesh_lat[antipodal_idx],
                          c='blue', s=20, transform=ccrs.PlateCarree())
        ax_global.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax_global.set_global()
        ax_global.set_title(f'Level {level}: Global View\n{len(vertices)} vertices')

        # 2. Regional view
        regional_extent = [
            center_lon - zoom_radius_deg,
            center_lon + zoom_radius_deg,
            center_lat - zoom_radius_deg * 0.7,
            center_lat + zoom_radius_deg * 0.7
        ]
        ax_regional.set_extent(regional_extent, crs=ccrs.PlateCarree())

        regional_mask = (
            (mesh_lon >= regional_extent[0]) &
            (mesh_lon <= regional_extent[1]) &
            (mesh_lat >= regional_extent[2]) &
            (mesh_lat <= regional_extent[3]) &
            is_regional
        )

        if np.any(regional_mask):
            ax_regional.scatter(mesh_lon[regional_mask], mesh_lat[regional_mask],
                                c='red', s=30, transform=ccrs.PlateCarree(), alpha=0.8)

        # Draw all edges that intersect regional extent
        for face in faces:
            face_lons = mesh_lon[face]
            face_lats = mesh_lat[face]

            if np.any(
                (face_lons >= regional_extent[0]) & (face_lons <= regional_extent[1]) &
                (face_lats >= regional_extent[2]) & (face_lats <= regional_extent[3])
            ):
                triangle_lons = np.append(face_lons, face_lons[0])
                triangle_lats = np.append(face_lats, face_lats[0])

                ax_regional.plot(triangle_lons, triangle_lats, 'b-', alpha=0.3,
                                 linewidth=0.5, transform=ccrs.PlateCarree())

        ax_regional.add_feature(cfeature.COASTLINE)
        ax_regional.add_feature(cfeature.BORDERS)
        ax_regional.add_feature(cfeature.OCEAN, alpha=0.3)
        ax_regional.add_feature(cfeature.LAND, alpha=0.3)
        ax_regional.gridlines(draw_labels=True, alpha=0.3)
        ax_regional.set_title(f'Level {level}: Regional Zoom')
        ax_regional.plot(center_lon, center_lat, 'g*', markersize=15, transform=ccrs.PlateCarree())

        # 3. Ultra close-up (10 degree box)
        close_extent = [
            center_lon - 10,
            center_lon + 10,
            center_lat - 10,
            center_lat + 10
        ]
        ax_close.set_extent(close_extent, crs=ccrs.PlateCarree())

        close_mask = (
            (mesh_lon >= close_extent[0]) &
            (mesh_lon <= close_extent[1]) &
            (mesh_lat >= close_extent[2]) &
            (mesh_lat <= close_extent[3]) &
            is_regional
        )

        if np.any(close_mask):
            ax_close.scatter(mesh_lon[close_mask], mesh_lat[close_mask],
                             c='red', s=50, transform=ccrs.PlateCarree())

            # Draw all edges that intersect close region (10° + 5° buffer)
            for face in faces:
                face_lons = mesh_lon[face]
                face_lats = mesh_lat[face]

                if np.any(
                    (face_lons >= close_extent[0] - 5) & (face_lons <= close_extent[1] + 5) &
                    (face_lats >= close_extent[2] - 5) & (face_lats <= close_extent[3] + 5)
                ):
                    triangle_lons = np.append(face_lons, face_lons[0])
                    triangle_lats = np.append(face_lats, face_lats[0])

                    ax_close.plot(triangle_lons, triangle_lats, 'b-',
                                  alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())

        ax_close.add_feature(cfeature.COASTLINE, linewidth=2)
        ax_close.add_feature(cfeature.BORDERS, linewidth=1)
        ax_close.add_feature(cfeature.OCEAN, alpha=0.3)
        ax_close.add_feature(cfeature.LAND, alpha=0.3)
        ax_close.gridlines(draw_labels=True)
        ax_close.set_title(f'Level {level}: Close-up Detail')
        ax_close.plot(center_lon, center_lat, 'g*', markersize=20, transform=ccrs.PlateCarree())

        # 4. 3D view of regional area
        regional_3d_mask = regional_mask
        if np.any(regional_3d_mask):
            regional_vertices = vertices[regional_3d_mask]
            center_3d = np.mean(regional_vertices, axis=0)
            regional_vertices_centered = regional_vertices - center_3d

            ax_3d_regional.scatter(
                regional_vertices_centered[:, 0],
                regional_vertices_centered[:, 1],
                regional_vertices_centered[:, 2],
                c='red', s=30, alpha=0.8
            )

            edges_drawn_3d = 0
            for face in faces:
                if np.all(regional_3d_mask[face]):
                    triangle = vertices[face] - center_3d
                    triangle = np.vstack([triangle, triangle[0]])
                    ax_3d_regional.plot(
                        triangle[:, 0], triangle[:, 1], triangle[:, 2],
                        'b-', alpha=0.3, linewidth=0.5
                    )
                    edges_drawn_3d += 1

        ax_3d_regional.set_xlabel('X')
        ax_3d_regional.set_ylabel('Y')
        ax_3d_regional.set_zlabel('Z')
        ax_3d_regional.set_title(f'Level {level}: 3D Regional')

        vertices_in_region = np.sum(close_mask)
        info_text = f"Vertices in ±10° box: {vertices_in_region}"
        fig.text(0.02, 0.90 - level * (1.0/n_levels), info_text,
                 fontsize=12, transform=fig.transFigure)

    plt.tight_layout()
    return fig


import xarray as xr
import pandas as pd
import numpy as np

def extend_forecast_to_target_times(ds, target_times):
    """
    Extend forecast data to match target times by interpolating/extending each forecast run.
    
    Args:
        ds: xarray Dataset with precipitation data
        target_times: array of target datetime64 values
        
    Returns:
        xarray Dataset with data at target times
    """
    
    # Convert target times to pandas datetime for easier manipulation
    target_times_pd = pd.to_datetime(target_times)
    
    print(f"Original times: {len(ds.time)} forecasts")
    print(f"Target times: {len(target_times)} time points")
    
    # Create new dataset with target times
    extended_data = []
    
    for target_time in target_times_pd:
        print(f"Processing target time: {target_time}")
        
        # Find the best forecast run for this target time
        best_forecast = None
        best_step = None
        min_lead_time = float('inf')
        
        for time_idx, forecast_time in enumerate(ds.time.values):
            forecast_time_pd = pd.to_datetime(forecast_time)
            
            # Calculate what step this target time would be for this forecast
            time_diff = target_time - forecast_time_pd
            hours_diff = time_diff.total_seconds() / 3600
            
            # Check if this forecast run covers this target time
            if hours_diff >= 0:  # Target time is after forecast start
                # Find the closest step in this forecast
                step_hours = []
                for step in ds.step.values:
                    step_seconds = pd.to_timedelta(step).total_seconds()
                    step_hours.append(step_seconds / 3600)
                
                step_hours = np.array(step_hours)
                
                # Find closest step
                closest_step_idx = np.argmin(np.abs(step_hours - hours_diff))
                closest_step_hours = step_hours[closest_step_idx]
                
                # Check if this step is close enough (within reasonable tolerance)
                if abs(closest_step_hours - hours_diff) <= 0.1:  # Within 6 minutes
                    lead_time = hours_diff
                    if lead_time < min_lead_time:
                        min_lead_time = lead_time
                        best_forecast = time_idx
                        best_step = closest_step_idx
        
        if best_forecast is not None:
            # Extract data from best forecast at best step
            data_slice = ds.tp.isel(time=best_forecast, step=best_step)
            print(f"  Using forecast {pd.to_datetime(ds.time.values[best_forecast])} step {best_step} (lead time: {min_lead_time:.1f}h)")
        else:
            # If no exact match, use interpolation or nearest neighbor
            print(f"  No exact match found, using nearest neighbor...")
            
            # Find the nearest forecast time
            forecast_times_pd = pd.to_datetime(ds.time.values)
            time_diffs = np.abs([(target_time - ft).total_seconds() for ft in forecast_times_pd])
            nearest_forecast_idx = np.argmin(time_diffs)
            
            # Use the first step from the nearest forecast
            data_slice = ds.tp.isel(time=nearest_forecast_idx, step=0)
            print(f"  Using nearest forecast {forecast_times_pd[nearest_forecast_idx]} step 0")
        
        extended_data.append(data_slice)
    
    # Combine all data slices
    extended_ds = xr.concat(extended_data, dim='new_time')
    extended_ds = extended_ds.assign_coords(new_time=target_times_pd)
    extended_ds = extended_ds.rename({'new_time': 'time'})
    
    return extended_ds

def create_6hourly_times(start_date, end_date):
    """
    Create 6-hourly time array (00, 06, 12, 18 UTC) for the given date range.
    """
    times = []
    current = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0, microsecond=0)
    end = pd.to_datetime(end_date)
    
    while current <= end:
        for hour in [0, 6, 12, 18]:
            time_point = current.replace(hour=hour)
            if time_point <= end:
                times.append(time_point)
        current += pd.Timedelta(days=1)
    
    return np.array(times, dtype='datetime64[ns]')

def extend_precipitation_data(grib_file_path, start_date, end_date):
    """
    Load precipitation data and extend it to 6-hourly intervals.
    """
    print(f"Loading precipitation data for {start_date} to {end_date}")
    
    # Load the data (using the method that avoids cfgrib conflicts)
    try:
        ds = xr.open_dataset(grib_file_path, engine='cfgrib')
        print(f"Loaded dataset with shape: {ds.tp.shape}")
    except Exception as e:
        print(f"Error loading with cfgrib: {e}")
        print("Trying to load forecast runs separately...")
        
        # Load first forecast run as example
        from cfgrib.messages import FileStream
        fs = FileStream(grib_file_path)
        first_msg = next(msg for msg in fs.items() if msg[1].get('shortName') == 'tp')
        date = first_msg[1].get('dataDate')
        time = first_msg[1].get('dataTime')
        
        ds = xr.open_dataset(
            grib_file_path, 
            engine='cfgrib',
            filter_by_keys={'dataDate': date, 'dataTime': time}
        )
        print(f"Loaded single forecast run with shape: {ds.tp.shape}")
    
    print(f"Original time range: {ds.time.values}")
    
    # Create target 6-hourly times
    target_times = create_6hourly_times(start_date, end_date)
    print(f"Target times: {target_times}")
    
    # Extend the data to match target times
    extended_ds = extend_forecast_to_target_times(ds, target_times)
    
    return extended_ds

# Quick function for your specific case
def quick_extend_times(ds_tp, target_times):
    """
    Quick extension of existing dataset to target times.
    """
    print("Current times:", ds_tp.time.values)
    print("Target times:", target_times)
    
    # Simple approach: for each target time, find the nearest available forecast
    extended_data = []
    
    for target_time in target_times:
        target_pd = pd.to_datetime(target_time)
        
        # Find nearest forecast time
        forecast_times = pd.to_datetime(ds_tp.time.values)
        time_diffs = np.abs([(target_pd - ft).total_seconds() for ft in forecast_times])
        nearest_idx = np.argmin(time_diffs)
        
        print(f"For {target_pd}, using forecast {forecast_times[nearest_idx]} (step 0)")
        
        # Take the first step from the nearest forecast
        data_slice = ds_tp.isel(time=nearest_idx, step=0)
        extended_data.append(data_slice)
    
    # Combine
    result = xr.concat(extended_data, dim='new_time')
    result = result.assign_coords(new_time=target_times)
    # result = result.rename({'new_time': 'time'})
    
    return result


def restructure_time_for_graphcast(ds):
    """
    Restructure time coordinates for GraphCast:
    - Rename current 'time' to 'datetime' 
    - Create new 'time' coordinate with timedelta64[ns]
    """
    print("Original coordinates:", list(ds.coords.keys()))
    print("Original time values:", ds.time.values)
    
    # Step 1: Rename 'time' to 'datetime'
    ds_renamed = ds.rename({'time': 'datetime'})
    
    # Step 2: Create new 'time' coordinate with timedelta64[ns]
    # This represents time steps from some reference point
    n_times = len(ds_renamed.datetime)
    
    # Create timedelta array (e.g., 0h, 6h, 12h, 18h for 6-hourly data)
    # Assuming 6-hourly data - adjust if different
    time_deltas = np.array([pd.Timedelta(hours=i*6) for i in range(n_times)], dtype='timedelta64[ns]')
    
    print(f"Created time deltas: {time_deltas}")
    
    # Step 3: Add the new 'time' coordinate
    ds_restructured = ds_renamed.assign_coords(time=('datetime', time_deltas))
    
    print("New coordinates:", list(ds_restructured.coords.keys()))
    print("New time values:", ds_restructured.time.values)
    print("Datetime values:", ds_restructured.datetime.values)
    
    return ds_restructured

def clean_and_restructure_for_graphcast(ds, select_ensemble_member=0):
    """
    Complete cleaning and restructuring for GraphCast:
    1. Remove forecast-specific coordinates
    2. Select ensemble member if needed
    3. Restructure time coordinates
    """
    print("=== CLEANING AND RESTRUCTURING FOR GRAPHCAST ===")
    
    # Step 1: Select ensemble member if 'number' dimension exists
    if 'number' in ds.dims:
        print(f"Selecting ensemble member {select_ensemble_member}")
        ds = ds.isel(number=select_ensemble_member)
    
    # Step 2: Remove forecast-specific coordinates
    coords_to_remove = ['step', 'number', 'surface']
    for coord in coords_to_remove:
        if coord in ds.coords:
            print(f"Removing coordinate: {coord}")
            ds = ds.drop_vars([coord], errors='ignore')
    
    # Step 3: Restructure time coordinates
    ds_restructured = restructure_time_for_graphcast(ds)
    
    return ds_restructured

def create_custom_time_deltas(datetime_values, reference_time=None):
    """
    Create custom time deltas based on actual datetime values.
    
    Args:
        datetime_values: Array of datetime64 values
        reference_time: Reference time (if None, uses first datetime)
    """
    if reference_time is None:
        reference_time = pd.to_datetime(datetime_values[0])
    else:
        reference_time = pd.to_datetime(reference_time)
    
    # Calculate time deltas from reference
    time_deltas = []
    for dt in datetime_values:
        dt_pd = pd.to_datetime(dt)
        delta = dt_pd - reference_time
        time_deltas.append(delta)
    
    return np.array(time_deltas, dtype='timedelta64[ns]')

def restructure_with_custom_reference(ds, reference_time=None):
    """
    Restructure time coordinates with a custom reference time.
    """
    print("=== RESTRUCTURING WITH CUSTOM REFERENCE ===")
    
    # Rename time to datetime
    ds_renamed = ds.rename({'time': 'datetime'})
    
    # Create custom time deltas
    time_deltas = create_custom_time_deltas(ds.time.values, reference_time)
    
    print(f"Reference time: {reference_time or pd.to_datetime(ds.time.values[0])}")
    print(f"Time deltas: {time_deltas}")
    
    # Add new time coordinate
    ds_restructured = ds_renamed.assign_coords(time=('datetime', time_deltas))
    
    return ds_restructured