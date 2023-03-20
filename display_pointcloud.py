import open3d as o3d
import numpy as np
from PIL import Image
import numpy as np
import scipy.io  
import sys

if __name__ == "__main__":

    filename = sys.argv[1]

    mat = scipy.io.loadmat(filename)
    index = np.random.randint(0,len(mat['pointcloud']))
    xyz = mat['pointcloud'][index][0]
    img = mat['image'][index]
    img = Image.fromarray((255*img).astype('uint8'),'RGB')
    img.show()
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Add color and estimate normals for better visualization.
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying Open3D pointcloud made using numpy array ...")
    o3d.visualization.draw([pcd])