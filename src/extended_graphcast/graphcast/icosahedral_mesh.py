# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for creating icosahedral meshes."""

import itertools
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.spatial import transform


class TriangularMesh(NamedTuple):
  """Data structure for triangular meshes.

  Attributes:
    vertices: spatial positions of the vertices of the mesh of shape
        [num_vertices, num_dims].
    faces: triangular faces of the mesh of shape [num_faces, 3]. Contains
        integer indices into `vertices`.

  """
  vertices: np.ndarray
  faces: np.ndarray


def merge_meshes(
    mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
  """Merges all meshes into one. Assumes the last mesh is the finest.

  Args:
     mesh_list: Sequence of meshes, from coarse to fine refinement levels. The
       vertices and faces may contain those from preceding, coarser levels.

  Returns:
     `TriangularMesh` for which the vertices correspond to the highest
     resolution mesh in the hierarchy, and the faces are the join set of the
     faces at all levels of the hierarchy.
  """
  for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
    num_nodes_mesh_i = mesh_i.vertices.shape[0]
    assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

  return TriangularMesh(
      vertices=mesh_list[-1].vertices,
      faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0))


def get_hierarchy_of_triangular_meshes_for_sphere(
    splits: int) -> List[TriangularMesh]:
  """Returns a sequence of meshes, each with triangularization sphere.

  Starting with a regular icosahedron (12 vertices, 20 faces, 30 edges) with
  circumscribed unit sphere. Then, each triangular face is iteratively
  subdivided into 4 triangular faces `splits` times. The new vertices are then
  projected back onto the unit sphere. All resulting meshes are returned in a
  list, from lowest to highest resolution.

  The vertices in each face are specified in counter-clockwise order as
  observed from the outside the icosahedron.

  Args:
     splits: How many times to split each triangle.
  Returns:
     Sequence of `TriangularMesh`s of length `splits + 1` each with:

       vertices: [num_vertices, 3] vertex positions in 3D, all with unit norm.
       faces: [num_faces, 3] with triangular faces joining sets of 3 vertices.
           Each row contains three indices into the vertices array, indicating
           the vertices adjacent to the face. Always with positive orientation
           (counterclock-wise when looking from the outside).
  """
  current_mesh = get_icosahedron()
  output_meshes = [current_mesh]
  for _ in range(splits):
    current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
    output_meshes.append(current_mesh)
  return output_meshes


def get_icosahedron() -> TriangularMesh:
  """Returns a regular icosahedral mesh with circumscribed unit sphere.

  See https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
  for details on the construction of the regular icosahedron.

  The vertices in each face are specified in counter-clockwise order as observed
  from the outside of the icosahedron.

  Returns:
     TriangularMesh with:

     vertices: [num_vertices=12, 3] vertex positions in 3D, all with unit norm.
     faces: [num_faces=20, 3] with triangular faces joining sets of 3 vertices.
         Each row contains three indices into the vertices array, indicating
         the vertices adjacent to the face. Always with positive orientation (
         counterclock-wise when looking from the outside).

  """
  phi = (1 + np.sqrt(5)) / 2
  vertices = []
  for c1 in [1., -1.]:
    for c2 in [phi, -phi]:
      vertices.append((c1, c2, 0.))
      vertices.append((0., c1, c2))
      vertices.append((c2, 0., c1))

  vertices = np.array(vertices, dtype=np.float32)
  vertices /= np.linalg.norm([1., phi])

  # I did this manually, checking the orientation one by one.
  faces = [(0, 1, 2),
           (0, 6, 1),
           (8, 0, 2),
           (8, 4, 0),
           (3, 8, 2),
           (3, 2, 7),
           (7, 2, 1),
           (0, 4, 6),
           (4, 11, 6),
           (6, 11, 5),
           (1, 5, 7),
           (4, 10, 11),
           (4, 8, 10),
           (10, 8, 3),
           (10, 3, 9),
           (11, 10, 9),
           (11, 9, 5),
           (5, 9, 7),
           (9, 3, 7),
           (1, 6, 5),
           ]

  # By default the top is an aris parallel to the Y axis.
  # Need to rotate around the y axis by half the supplementary to the
  # angle between faces divided by two to get the desired orientation.
  #                          /O\  (top arist)
  #                     /          \                           Z
  # (adjacent face)/                    \  (adjacent face)     ^
  #           /     angle_between_faces      \                 |
  #      /                                        \            |
  #  /                                                 \      YO-----> X
  # This results in:
  #  (adjacent faceis now top plane)
  #  ----------------------O\  (top arist)
  #                           \
  #                             \
  #                               \     (adjacent face)
  #                                 \
  #                                   \
  #                                     \

  angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
  rotation_angle = (np.pi - angle_between_faces) / 2
  rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
  rotation_matrix = rotation.as_matrix()
  vertices = np.dot(vertices, rotation_matrix)

  return TriangularMesh(vertices=vertices.astype(np.float32),
                        faces=np.array(faces, dtype=np.int32))


def _two_split_unit_sphere_triangle_faces(
    triangular_mesh: TriangularMesh) -> TriangularMesh:
  """Splits each triangular face into 4 triangles keeping the orientation."""

  # Every time we split a triangle into 4 we will be adding 3 extra vertices,
  # located at the edge centres.
  # This class handles the positioning of the new vertices, and avoids creating
  # duplicates.
  new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

  new_faces = []
  for ind1, ind2, ind3 in triangular_mesh.faces:
    # Transform each triangular face into 4 triangles,
    # preserving the orientation.
    #                    ind3
    #                   /    \
    #                /          \
    #              /      #3       \
    #            /                  \
    #         ind31 -------------- ind23
    #         /   \                /   \
    #       /       \     #4     /      \
    #     /    #1     \        /    #2    \
    #   /               \    /              \
    # ind1 ------------ ind12 ------------ ind2
    ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
    ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
    ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
    # Note how each of the 4 triangular new faces specifies the order of the
    # vertices to preserve the orientation of the original face. As the input
    # face should always be counter-clockwise as specified in the diagram,
    # this means child faces should also be counter-clockwise.
    new_faces.extend([[ind1, ind12, ind31],  # 1
                      [ind12, ind2, ind23],  # 2
                      [ind31, ind23, ind3],  # 3
                      [ind12, ind23, ind31],  # 4
                      ])
  return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(),
                        faces=np.array(new_faces, dtype=np.int32))


class _ChildVerticesBuilder(object):
  """Bookkeeping of new child vertices added to an existing set of vertices."""

  def __init__(self, parent_vertices):

    # Because the same new vertex will be required when splitting adjacent
    # triangles (which share an edge) we keep them in a hash table indexed by
    # sorted indices of the vertices adjacent to the edge, to avoid creating
    # duplicated child vertices.
    self._child_vertices_index_mapping = {}
    self._parent_vertices = parent_vertices
    # We start with all previous vertices.
    self._all_vertices_list = list(parent_vertices)

  def _get_child_vertex_key(self, parent_vertex_indices):
    return tuple(sorted(parent_vertex_indices))

  def _create_child_vertex(self, parent_vertex_indices):
    """Creates a new vertex."""
    # Position for new vertex is the middle point, between the parent points,
    # projected to unit sphere.
    child_vertex_position = self._parent_vertices[
        list(parent_vertex_indices)].mean(0)
    child_vertex_position /= np.linalg.norm(child_vertex_position)

    # Add the vertex to the output list. The index for this new vertex will
    # match the length of the list before adding it.
    child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
    self._child_vertices_index_mapping[child_vertex_key] = len(
        self._all_vertices_list)
    self._all_vertices_list.append(child_vertex_position)

  def get_new_child_vertex_index(self, parent_vertex_indices):
    """Returns index for a child vertex, creating it if necessary."""
    # Get the key to see if we already have a new vertex in the middle.
    child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
    if child_vertex_key not in self._child_vertices_index_mapping:
      self._create_child_vertex(parent_vertex_indices)
    return self._child_vertices_index_mapping[child_vertex_key]

  def get_all_vertices(self):
    """Returns an array with old vertices."""
    return np.array(self._all_vertices_list)


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms polygonal faces to sender and receiver indices.

  It does so by transforming every face into N_i edges. Such if the triangular
  face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

  If all faces have consistent orientation, and the surface represented by the
  faces is closed, then every edge in a polygon with a certain orientation
  is also part of another polygon with the opposite orientation. In this
  situation, the edges returned by the method are always bidirectional.

  Args:
    faces: Integer array of shape [num_faces, 3]. Contains node indices
        adjacent to each face.
  Returns:
    Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

  """
  assert faces.ndim == 2
  assert faces.shape[-1] == 3
  senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
  receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
  return senders, receivers


def get_last_triangular_mesh_for_sphere(splits: int) -> TriangularMesh:
  return get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)[-1]

def lat_lon_to_cartesian(lat: float, lon: float, radius: float = 1.0) -> np.ndarray:
    """Convert latitude/longitude to 3D cartesian coordinates."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return np.array([x, y, z])

def get_regional_focused_icosahedron(
    center_lat: float, 
    center_lon: float,
    regional_radius_deg: float = 30.0) -> TriangularMesh:
    """Create an icosahedron with 11 vertices concentrated around a region.
    
    Args:
        center_lat: Latitude of the center point (degrees)
        center_lon: Longitude of the center point (degrees)
        regional_radius_deg: Approximate radius of the region in degrees
    
    Returns:
        TriangularMesh with 12 vertices focused on the region
    """
    
    vertices = []
    
    # 1. Center point of the region of interest
    center_point = lat_lon_to_cartesian(center_lat, center_lon)
    vertices.append(center_point)
    
    # 2. Create 10 points around the center in a pentagonal bipyramid pattern
    # This maintains some of the icosahedral structure
    
    # Convert regional radius to approximate 3D distance
    radius_3d = np.sin(np.deg2rad(regional_radius_deg))
    
    # Create two rings of 5 points each
    for ring_idx, height_factor in enumerate([0.5, -0.5]):
        # Height offset from the center point
        height_offset = height_factor * radius_3d * 0.5
        
        # 5 points in a pentagon around the center
        for i in range(5):
            angle = 2 * np.pi * i / 5 + (ring_idx * np.pi / 5)  # Offset second ring
            
            # Create point in local tangent plane
            local_east = np.array([-center_point[1], center_point[0], 0])
            local_east /= np.linalg.norm(local_east)
            local_north = np.cross(center_point, local_east)
            
            # Point position relative to center
            offset = (radius_3d * np.cos(angle) * local_east + 
                     radius_3d * np.sin(angle) * local_north +
                     height_offset * center_point)
            
            point = center_point + offset
            point /= np.linalg.norm(point)  # Project back to unit sphere
            vertices.append(point)
    
    # 3. Antipodal point (opposite side of Earth)
    antipodal_point = -center_point
    vertices.append(antipodal_point)
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Define faces to create a closed mesh
    # Center point (0) connects to first ring (1-5)
    # First ring connects to second ring (6-10)
    # Second ring connects to antipodal point (11)
    
    faces = []
    
    # Connect center to first ring
    for i in range(5):
        next_i = (i + 1) % 5
        faces.append([0, i + 1, next_i + 1])
    
    # Connect first ring to second ring
    for i in range(5):
        next_i = (i + 1) % 5
        faces.append([i + 1, next_i + 1, i + 6])
        faces.append([next_i + 1, next_i + 6, i + 6])
    
    # Connect second ring to antipodal point
    for i in range(5):
        next_i = (i + 1) % 5
        faces.append([i + 6, next_i + 6, 11])
    
    faces = np.array(faces, dtype=np.int32)
    
    return TriangularMesh(vertices=vertices, faces=faces)



def _eight_split_unit_sphere_triangle_faces_regional(
    triangular_mesh: TriangularMesh,
    antipodal_vertex_idx: int = None) -> TriangularMesh:
    """Splits each triangular face into 8 triangles, excluding faces touching the antipodal vertex.
    
    The 8-way split creates more vertices per face:
    - 3 vertices at edge midpoints (like 4-way split)
    - 3 vertices at 1/3 and 2/3 points along each edge
    - 1 vertex at the centroid
    """
    
    # Find all faces that include the antipodal vertex
    faces_to_exclude = set()
    if antipodal_vertex_idx is not None:
        for i, face in enumerate(triangular_mesh.faces):
            if antipodal_vertex_idx in face:
                faces_to_exclude.add(i)
    
    new_vertices_builder = _RegionalChildVerticesBuilder(triangular_mesh.vertices)
    new_faces = []
    
    for face_idx, (ind1, ind2, ind3) in enumerate(triangular_mesh.faces):
        # Skip faces connected to antipodal vertex
        if face_idx in faces_to_exclude:
            new_faces.append([ind1, ind2, ind3])  # Keep original face
            continue
        
        # For 8-way split, we need:
        # - Edge midpoints (3)
        # - Edge third-points (6) 
        # - Face centroid (1)
        
        # Edge midpoints
        ind12_mid = new_vertices_builder.get_new_child_vertex_index((ind1, ind2), fraction=0.5)
        ind23_mid = new_vertices_builder.get_new_child_vertex_index((ind2, ind3), fraction=0.5)
        ind31_mid = new_vertices_builder.get_new_child_vertex_index((ind3, ind1), fraction=0.5)
        
        # Edge third-points (1/3 and 2/3 along each edge)
        ind12_1third = new_vertices_builder.get_new_child_vertex_index((ind1, ind2), fraction=1/3)
        ind12_2third = new_vertices_builder.get_new_child_vertex_index((ind1, ind2), fraction=2/3)
        ind23_1third = new_vertices_builder.get_new_child_vertex_index((ind2, ind3), fraction=1/3)
        ind23_2third = new_vertices_builder.get_new_child_vertex_index((ind2, ind3), fraction=2/3)
        ind31_1third = new_vertices_builder.get_new_child_vertex_index((ind3, ind1), fraction=1/3)
        ind31_2third = new_vertices_builder.get_new_child_vertex_index((ind3, ind1), fraction=2/3)
        
        # Face centroid
        ind_centroid = new_vertices_builder.get_new_face_centroid_index((ind1, ind2, ind3))
        
        # Create 8 triangular faces
        # This is the subdivision pattern:
        #                    ind3
        #                   /    \
        #                 /        \
        #           ind31_2third   ind23_2third
        #              /              \
        #            /                  \
        #      ind31_mid    centroid    ind23_mid
        #          /                      \
        #        /                          \
        #  ind31_1third                  ind23_1third
        #      /                              \
        #    /                                  \
        # ind1--ind12_1third--ind12_mid--ind12_2third--ind2
        
        # Corner triangles
        new_faces.extend([
            [ind1, ind12_1third, ind31_1third],
            [ind2, ind23_1third, ind12_2third],
            [ind3, ind31_2third, ind23_2third],
        ])
        
        # Edge triangles
        new_faces.extend([
            [ind12_1third, ind12_mid, ind_centroid],
            [ind12_mid, ind12_2third, ind_centroid],
            [ind23_1third, ind23_mid, ind_centroid],
            [ind23_mid, ind23_2third, ind_centroid],
            [ind31_1third, ind31_mid, ind_centroid],
            [ind31_mid, ind31_2third, ind_centroid],
        ])
        
        # Connect remaining pieces
        new_faces.extend([
            [ind31_1third, ind_centroid, ind12_1third],
            [ind12_2third, ind_centroid, ind23_1third],
            [ind23_2third, ind_centroid, ind31_2third],
        ])
    
    return TriangularMesh(
        vertices=new_vertices_builder.get_all_vertices(),
        faces=np.array(new_faces, dtype=np.int32)
    )


class _RegionalChildVerticesBuilder(_ChildVerticesBuilder):
    """Extended vertex builder that supports fractional positions along edges."""
    
    def __init__(self, parent_vertices):
        super().__init__(parent_vertices)
        self._face_centroid_index_mapping = {}
    
    def _get_child_vertex_key_fractional(self, parent_vertex_indices, fraction):
        """Create unique key for vertices at specific fractions along edges."""
        sorted_indices = tuple(sorted(parent_vertex_indices))
        # Include fraction in the key to distinguish 1/3, 1/2, 2/3 points
        return (*sorted_indices, fraction)
    
    def get_new_child_vertex_index(self, parent_vertex_indices, fraction=0.5):
        """Get vertex at specified fraction along edge between parent vertices."""
        key = self._get_child_vertex_key_fractional(parent_vertex_indices, fraction)
        
        if key not in self._child_vertices_index_mapping:
            # Create vertex at specified fraction
            v1_idx, v2_idx = parent_vertex_indices
            v1 = self._parent_vertices[v1_idx]
            v2 = self._parent_vertices[v2_idx]
            
            # Linear interpolation
            child_vertex_position = (1 - fraction) * v1 + fraction * v2
            # Project to unit sphere
            child_vertex_position /= np.linalg.norm(child_vertex_position)
            
            # Add to vertices
            self._child_vertices_index_mapping[key] = len(self._all_vertices_list)
            self._all_vertices_list.append(child_vertex_position)
        
        return self._child_vertices_index_mapping[key]
    
    def get_new_face_centroid_index(self, face_indices):
        """Get vertex at the centroid of a triangular face."""
        key = tuple(sorted(face_indices))
        
        if key not in self._face_centroid_index_mapping:
            # Create centroid vertex
            vertices = self._parent_vertices[list(face_indices)]
            centroid_position = vertices.mean(axis=0)
            # Project to unit sphere
            centroid_position /= np.linalg.norm(centroid_position)
            
            # Add to vertices
            self._face_centroid_index_mapping[key] = len(self._all_vertices_list)
            self._all_vertices_list.append(centroid_position)
        
        return self._face_centroid_index_mapping[key]


def get_regional_hierarchy_with_8way_split(
    splits: int,
    center_lat: float,
    center_lon: float,
    regional_radius_deg: float = 30.0) -> List[TriangularMesh]:
    """Create mesh hierarchy with 8-way subdivision focused on a region.
    
    The antipodal vertex (last vertex, index 11) remains isolated - no new
    triangles are created using it.
    """
    
    # Start with regional-focused icosahedron
    # current_mesh = get_regional_focused_icosahedron(
    #     center_lat, center_lon, regional_radius_deg
    # )
    current_mesh = get_regional_radial_mesh(center_lat, center_lon)
    
    output_meshes = [current_mesh]
    antipodal_idx = len(current_mesh.vertices) - 1  # Last vertex is antipodal
    
    for split_level in range(splits):
        print(f"Split level {split_level}: {len(current_mesh.vertices)} vertices, {len(current_mesh.faces)} faces")
        
        # # Apply 8-way subdivision, keeping antipodal vertex isolated
        # current_mesh = _sixteen_split_unit_sphere_triangle_faces_regional( # _eight_split_unit_sphere_triangle_faces_regional
        #     current_mesh, 
        #     antipodal_vertex_idx=antipodal_idx
        # )
        # current_mesh = _adaptive_split_unit_sphere_triangle_faces_regional(
        #     current_mesh,
        #     center_point=lat_lon_to_cartesian(center_lat, center_lon),
        #     antipodal_vertex_idx=antipodal_idx,
        #     split_level=split_level * 10
        # )

        current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)

        output_meshes.append(current_mesh)
        
        # Antipodal vertex index remains the same (no new vertices before it)
        # since we're not subdividing faces connected to it
    
    return output_meshes


# Simplified 4-way version that also excludes antipodal vertex
def _four_split_excluding_antipode(
    triangular_mesh: TriangularMesh,
    antipodal_vertex_idx: int) -> TriangularMesh:
    """Standard 4-way split but excluding faces with antipodal vertex."""
    
    faces_to_exclude = set()
    for i, face in enumerate(triangular_mesh.faces):
        if antipodal_vertex_idx in face:
            faces_to_exclude.add(i)
    
    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)
    new_faces = []
    
    for face_idx, (ind1, ind2, ind3) in enumerate(triangular_mesh.faces):
        if face_idx in faces_to_exclude:
            new_faces.append([ind1, ind2, ind3])  # Keep original
            continue
        
        # Standard 4-way subdivision
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
        
        new_faces.extend([
            [ind1, ind12, ind31],
            [ind12, ind2, ind23],
            [ind31, ind23, ind3],
            [ind12, ind23, ind31],
        ])
    
    return TriangularMesh(
        vertices=new_vertices_builder.get_all_vertices(),
        faces=np.array(new_faces, dtype=np.int32)
    )




def _sixteen_split_unit_sphere_triangle_faces_regional(
    triangular_mesh: TriangularMesh,
    antipodal_vertex_idx: int = None) -> TriangularMesh:
    """Splits each triangular face into 16 smaller triangles (creating 17 total regions).
    
    The pattern creates 4 "rows" of triangles:
    - Row 1 (top): 1 triangle
    - Row 2: 3 triangles  
    - Row 3: 5 triangles
    - Row 4 (bottom): 7 triangles
    Total: 1 + 3 + 5 + 7 = 16 triangles
    """
    
    # Find faces to exclude (connected to antipodal vertex)
    faces_to_exclude = set()
    if antipodal_vertex_idx is not None:
        for i, face in enumerate(triangular_mesh.faces):
            if antipodal_vertex_idx in face:
                faces_to_exclude.add(i)
    
    new_vertices_builder = _ExtendedChildVerticesBuilder(triangular_mesh.vertices)
    new_faces = []
    
    for face_idx, (ind1, ind2, ind3) in enumerate(triangular_mesh.faces):
        # Skip faces connected to antipodal vertex
        if face_idx in faces_to_exclude:
            new_faces.append([ind1, ind2, ind3])
            continue
        
        # For 16-way split, we need vertices at 1/4, 1/2, 3/4 positions on each edge
        # This creates a 4x4 grid of vertices within the triangle
        
        # Edge 1-2 vertices
        ind12_quarter = new_vertices_builder.get_new_child_vertex_index((ind1, ind2), fraction=0.25)
        ind12_half = new_vertices_builder.get_new_child_vertex_index((ind1, ind2), fraction=0.5)
        ind12_three_quarter = new_vertices_builder.get_new_child_vertex_index((ind1, ind2), fraction=0.75)
        
        # Edge 2-3 vertices  
        ind23_quarter = new_vertices_builder.get_new_child_vertex_index((ind2, ind3), fraction=0.25)
        ind23_half = new_vertices_builder.get_new_child_vertex_index((ind2, ind3), fraction=0.5)
        ind23_three_quarter = new_vertices_builder.get_new_child_vertex_index((ind2, ind3), fraction=0.75)
        
        # Edge 3-1 vertices
        ind31_quarter = new_vertices_builder.get_new_child_vertex_index((ind3, ind1), fraction=0.25)
        ind31_half = new_vertices_builder.get_new_child_vertex_index((ind3, ind1), fraction=0.5)
        ind31_three_quarter = new_vertices_builder.get_new_child_vertex_index((ind3, ind1), fraction=0.75)
        
        # Interior vertices - we need to create a grid inside the triangle
        # These are at barycentric coordinates
        interior_vertices = []
        
        # Row 2 interior vertex (1 vertex)
        v_row2 = new_vertices_builder.get_interior_vertex_index(
            (ind1, ind2, ind3), barycentric=(1/4, 1/4, 1/2))
        
        # Row 3 interior vertices (2 vertices)
        v_row3_1 = new_vertices_builder.get_interior_vertex_index(
            (ind1, ind2, ind3), barycentric=(1/4, 1/2, 1/4))
        v_row3_2 = new_vertices_builder.get_interior_vertex_index(
            (ind1, ind2, ind3), barycentric=(1/2, 1/4, 1/4))
        
        # Now create the 16 triangles
        # The pattern looks like:
        #           ind3
        #            /\
        #           /  \
        #          / 1  \
        #         /______\
        #        /\  2  /\
        #       /  \  /  \
        #      / 3  \/  4 \
        #     /______\____\
        #    /\  5  /\  6 /\
        #   /  \  /  \  /  \
        #  / 7  \/ 8  \/ 9  \
        # /______\____\____\
        # ind1              ind2
        
        # Row 1 (top) - 1 triangle
        new_faces.append([ind3, ind31_three_quarter, ind23_three_quarter])
        
        # Row 2 - 3 triangles
        new_faces.extend([
            [ind31_three_quarter, ind31_half, v_row2],
            [ind31_three_quarter, v_row2, ind23_three_quarter],
            [ind23_three_quarter, v_row2, ind23_half],
        ])
        
        # Row 3 - 5 triangles
        new_faces.extend([
            [ind31_half, ind31_quarter, v_row3_1],
            [ind31_half, v_row3_1, v_row2],
            [v_row2, v_row3_1, v_row3_2],
            [v_row2, v_row3_2, ind23_half],
            [ind23_half, v_row3_2, ind23_quarter],
        ])
        
        # Row 4 (bottom) - 7 triangles
        new_faces.extend([
            [ind1, ind12_quarter, ind31_quarter],
            [ind31_quarter, ind12_quarter, v_row3_1],
            [ind12_quarter, ind12_half, v_row3_1],
            [v_row3_1, ind12_half, v_row3_2],
            [ind12_half, ind12_three_quarter, v_row3_2],
            [v_row3_2, ind12_three_quarter, ind23_quarter],
            [ind12_three_quarter, ind2, ind23_quarter],
        ])
    
    return TriangularMesh(
        vertices=new_vertices_builder.get_all_vertices(),
        faces=np.array(new_faces, dtype=np.int32)
    )


class _ExtendedChildVerticesBuilder:
    """Vertex builder that supports fractional edge positions and interior vertices."""
    
    def __init__(self, parent_vertices):
        self._parent_vertices = parent_vertices
        self._all_vertices_list = list(parent_vertices)
        self._edge_vertex_index_mapping = {}
        self._interior_vertex_index_mapping = {}
    
    def get_new_child_vertex_index(self, parent_vertex_indices, fraction=0.5):
        """Get vertex at specified fraction along edge between parent vertices."""
        # Create unique key
        v1_idx, v2_idx = parent_vertex_indices
        if v1_idx > v2_idx:
            v1_idx, v2_idx = v2_idx, v1_idx
        key = (v1_idx, v2_idx, fraction)
        
        if key not in self._edge_vertex_index_mapping:
            # Create vertex
            v1 = self._parent_vertices[v1_idx]
            v2 = self._parent_vertices[v2_idx]
            
            # Linear interpolation
            new_vertex = (1 - fraction) * v1 + fraction * v2
            # Project to unit sphere
            new_vertex /= np.linalg.norm(new_vertex)
            
            # Add to vertices
            self._edge_vertex_index_mapping[key] = len(self._all_vertices_list)
            self._all_vertices_list.append(new_vertex)
        
        return self._edge_vertex_index_mapping[key]
    
    def get_interior_vertex_index(self, triangle_indices, barycentric):
        """Get vertex at specified barycentric coordinates within triangle."""
        # Create unique key
        sorted_indices = tuple(sorted(triangle_indices))
        key = (*sorted_indices, *barycentric)
        
        if key not in self._interior_vertex_index_mapping:
            # Create vertex using barycentric coordinates
            v1, v2, v3 = [self._parent_vertices[i] for i in triangle_indices]
            b1, b2, b3 = barycentric
            
            new_vertex = b1 * v1 + b2 * v2 + b3 * v3
            # Project to unit sphere
            new_vertex /= np.linalg.norm(new_vertex)
            
            # Add to vertices
            self._interior_vertex_index_mapping[key] = len(self._all_vertices_list)
            self._all_vertices_list.append(new_vertex)
        
        return self._interior_vertex_index_mapping[key]
    
    def get_all_vertices(self):
        return np.array(self._all_vertices_list, dtype=np.float32)


def visualize_subdivision_pattern():
    """Visualize the 16-way subdivision pattern on a single triangle."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original triangle
    triangle = patches.Polygon([(0, 0), (1, 0), (0.5, 0.866)], 
                              fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(triangle)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Original Triangle')
    
    # 16-way subdivision
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1)
    ax2.set_aspect('equal')
    ax2.set_title('16-way Subdivision (16 triangles)')
    
    # Draw the subdivision grid
    # Horizontal lines
    for i in range(1, 4):
        y = i * 0.866 / 4
        x_start = i / 8
        x_end = 1 - i / 8
        ax2.plot([x_start, x_end], [y, y], 'b-', alpha=0.5)
    
    # Left diagonal lines
    for i in range(1, 4):
        ax2.plot([0, 0.5 - i/8], [0, i * 0.866/4], 'b-', alpha=0.5)
    
    # Right diagonal lines  
    for i in range(1, 4):
        ax2.plot([1, 0.5 + i/8], [0, i * 0.866/4], 'b-', alpha=0.5)
    
    # Outer triangle
    ax2.plot([0, 1, 0.5, 0], [0, 0, 0.866, 0], 'k-', linewidth=2)
    
    # Number the triangles
    positions = [
        (0.5, 0.7),      # 1
        (0.3, 0.5),      # 2
        (0.5, 0.5),      # 3
        (0.7, 0.5),      # 4
        (0.2, 0.3),      # 5
        (0.35, 0.3),     # 6
        (0.5, 0.3),      # 7
        (0.65, 0.3),     # 8
        (0.8, 0.3),      # 9
        (0.1, 0.1),      # 10
        (0.25, 0.1),     # 11
        (0.35, 0.1),     # 12
        (0.5, 0.1),      # 13
        (0.65, 0.1),     # 14
        (0.75, 0.1),     # 15
        (0.9, 0.1),      # 16
    ]
    
    for i, (x, y) in enumerate(positions):
        ax2.text(x, y, str(i+1), ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig



def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angular distance between two points on unit sphere."""
    # Clip to handle numerical errors
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot_product)

def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angular distance between two points on unit sphere."""
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot_product)


def get_regional_hierarchy_with_adaptive_split(
    splits: int,
    center_lat: float,
    center_lon: float,
    regional_radius_deg: float = 30.0) -> List[TriangularMesh]:
    """Create mesh hierarchy with ADAPTIVE subdivision focused on a region.
    
    Uses distance-based refinement:
    - Very close to center: 16-way split
    - Close: 8-way split
    - Medium: 4-way split
    - Far: no split
    """
    
    # Start with regional-focused icosahedron (your existing function)
    current_mesh = get_regional_focused_icosahedron(
        center_lat, center_lon, regional_radius_deg
    )
    
    output_meshes = [current_mesh]
    antipodal_idx = len(current_mesh.vertices) - 1  # Last vertex is antipodal
    
    # Store the center point for distance calculations
    center_point = lat_lon_to_cartesian(center_lat, center_lon)
    
    for split_level in range(splits):
        print(f"Split level {split_level}: {len(current_mesh.vertices)} vertices, {len(current_mesh.faces)} faces")
        
        # Apply ADAPTIVE subdivision based on distance from center
        current_mesh = _adaptive_split_unit_sphere_triangle_faces_regional(
            current_mesh, 
            center_point=center_point,
            antipodal_vertex_idx=antipodal_idx,
            split_level=split_level  # Can adjust thresholds based on level
        )
        output_meshes.append(current_mesh)
    
    return output_meshes


def _adaptive_split_unit_sphere_triangle_faces_regional(
    triangular_mesh: TriangularMesh,
    center_point: np.ndarray,
    antipodal_vertex_idx: int = None,
    split_level: int = 0) -> TriangularMesh:
    """Adaptively split faces based on distance from center point.
    
    Distance thresholds can be adjusted based on split_level to gradually
    expand the high-resolution region.
    """
    
    # Find faces to exclude (connected to antipodal vertex)
    faces_to_exclude = set()
    if antipodal_vertex_idx is not None:
        for i, face in enumerate(triangular_mesh.faces):
            if antipodal_vertex_idx in face:
                faces_to_exclude.add(i)
    
    # Adjust distance thresholds based on split level
    # As we go deeper, we can be more selective about what gets refined
    base_very_close = 15  # degrees
    base_close = 30
    base_medium = 50
    
    # Make thresholds tighter at higher levels
    level_factor = 0.8 ** split_level
    very_close_threshold = np.deg2rad(base_very_close * level_factor)
    close_threshold = np.deg2rad(base_close * level_factor)
    medium_threshold = np.deg2rad(base_medium * level_factor)
    
    new_vertices_builder = _ExtendedChildVerticesBuilder(triangular_mesh.vertices)
    new_faces = []
    
    # Statistics for this level
    splits_16way = 0
    splits_8way = 0
    splits_4way = 0
    no_splits = 0
    
    for face_idx, face in enumerate(triangular_mesh.faces):
        # Skip faces connected to antipodal vertex
        if face_idx in faces_to_exclude:
            new_faces.append(face.tolist())
            no_splits += 1
            continue
        
        # Calculate face center distance from point of interest
        face_vertices = triangular_mesh.vertices[face]
        face_center = np.mean(face_vertices, axis=0)
        face_center /= np.linalg.norm(face_center)
        
        distance = angular_distance(face_center, center_point)
        
        # Choose subdivision based on distance
        if distance < very_close_threshold:
            # 16-way split for very close faces
            new_faces.extend(_do_16way_split(face, new_vertices_builder))
            splits_16way += 1
            
        elif distance < close_threshold:
            # 8-way split for close faces
            new_faces.extend(_do_8way_split(face, new_vertices_builder))
            splits_8way += 1
            
        elif distance < medium_threshold:
            # 4-way split for medium distance faces
            new_faces.extend(_do_4way_split(face, new_vertices_builder))
            splits_4way += 1
            
        else:
            # No split for far faces
            new_faces.append(face.tolist())
            no_splits += 1
    
    print(f"  Splits: 16-way={splits_16way}, 8-way={splits_8way}, "
          f"4-way={splits_4way}, none={no_splits}")
    
    return TriangularMesh(
        vertices=new_vertices_builder.get_all_vertices(),
        faces=np.array(new_faces, dtype=np.int32)
    )


def _do_4way_split(face: np.ndarray, builder) -> List[List[int]]:
    """Standard 4-way split."""
    ind1, ind2, ind3 = face
    
    # Get midpoints
    ind12 = builder.get_new_child_vertex_index((ind1, ind2), 0.5)
    ind23 = builder.get_new_child_vertex_index((ind2, ind3), 0.5)
    ind31 = builder.get_new_child_vertex_index((ind3, ind1), 0.5)
    
    return [
        [ind1, ind12, ind31],
        [ind12, ind2, ind23],
        [ind31, ind23, ind3],
        [ind12, ind23, ind31]
    ]


def _do_8way_split(face: np.ndarray, builder) -> List[List[int]]:
    """8-way split (you already have this implemented)."""
    ind1, ind2, ind3 = face
    
    # Edge midpoints
    ind12_mid = builder.get_new_child_vertex_index((ind1, ind2), 0.5)
    ind23_mid = builder.get_new_child_vertex_index((ind2, ind3), 0.5)
    ind31_mid = builder.get_new_child_vertex_index((ind3, ind1), 0.5)
    
    # Edge third points
    ind12_third1 = builder.get_new_child_vertex_index((ind1, ind2), 1/3)
    ind12_third2 = builder.get_new_child_vertex_index((ind1, ind2), 2/3)
    ind23_third1 = builder.get_new_child_vertex_index((ind2, ind3), 1/3)
    ind23_third2 = builder.get_new_child_vertex_index((ind2, ind3), 2/3)
    ind31_third1 = builder.get_new_child_vertex_index((ind3, ind1), 1/3)
    ind31_third2 = builder.get_new_child_vertex_index((ind3, ind1), 2/3)
    
    # Centroid
    centroid = builder.get_interior_vertex_index((ind1, ind2, ind3), (1/3, 1/3, 1/3))
    
    # Return 8 faces (abbreviated for space - use your existing pattern)
    return [
        # ... your 8-way split pattern
    ]


def _do_16way_split(face: np.ndarray, builder) -> List[List[int]]:
    """16-way split (you already have this implemented)."""
    # Use your existing _sixteen_split implementation
    # Just extract the face creation part
    ind1, ind2, ind3 = face
    
    # ... (use your existing 16-way split code)
    
    return [
        # ... your 16 faces
    ]


# Alternative: Create a radial initial mesh instead of icosahedral
def get_regional_radial_mesh(
    center_lat: float,
    center_lon: float,
    num_rings: int = 3) -> TriangularMesh:
    """Create initial mesh with radial rings centered on region.
    
    This naturally concentrates vertices near the center.
    """
    vertices = []
    center_point = lat_lon_to_cartesian(center_lat, center_lon)
    
    # Center point
    vertices.append(center_point)
    
    # Create concentric rings
    ring_radii = [8, 16, 25, 35, 50]  # degrees
    vertices_per_ring = [6, 12, 18, 24, 30]
    
    ring_indices = [[0]]  # Center is ring 0
    
    for ring_idx in range(min(num_rings, len(ring_radii))):
        radius_deg = ring_radii[ring_idx]
        num_vertices = vertices_per_ring[ring_idx]
        radius_3d = np.sin(np.deg2rad(radius_deg))
        
        ring_vertices = []
        
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            
            # Create local coordinate system
            local_east = np.array([-center_point[1], center_point[0], 0])
            local_east /= np.linalg.norm(local_east)
            local_north = np.cross(center_point, local_east)
            
            # Create vertex
            offset = radius_3d * (np.cos(angle) * local_east + np.sin(angle) * local_north)
            vertex = center_point + offset
            vertex /= np.linalg.norm(vertex)
            
            vertices.append(vertex)
            ring_vertices.append(len(vertices) - 1)
        
        ring_indices.append(ring_vertices)
    
    # Antipodal point
    vertices.append(-center_point)
    antipodal_idx = len(vertices) - 1
    
    # Create faces
    faces = []
    
    # Connect center to first ring
    first_ring = ring_indices[1]
    for i in range(len(first_ring)):
        next_i = (i + 1) % len(first_ring)
        faces.append([0, first_ring[i], first_ring[next_i]])
    
    # Connect consecutive rings
    for r in range(1, len(ring_indices) - 1):
        inner_ring = ring_indices[r]
        outer_ring = ring_indices[r + 1]
        
        # Simple connection pattern
        ratio = len(outer_ring) / len(inner_ring)
        
        for i in range(len(inner_ring)):
            # Connect each inner vertex to ~ratio outer vertices
            for j in range(int(i * ratio), int((i + 1) * ratio)):
                j_wrap = j % len(outer_ring)
                next_j = (j + 1) % len(outer_ring)
                next_i = (i + 1) % len(inner_ring)
                
                faces.append([inner_ring[i], outer_ring[j_wrap], outer_ring[next_j]])
                if j < int((i + 1) * ratio) - 1:
                    faces.append([inner_ring[i], outer_ring[next_j], inner_ring[next_i]])
    
    # Connect outermost ring to antipodal
    outer_ring = ring_indices[-1]
    for i in range(len(outer_ring)):
        next_i = (i + 1) % len(outer_ring)
        faces.append([outer_ring[i], outer_ring[next_i], antipodal_idx])
    
    return TriangularMesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32)
    )