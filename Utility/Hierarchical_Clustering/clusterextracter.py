from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from HC import Hierarchical_Clastering


# The ConnectedCluster class is used to find connected components in a graph represented by a list of
# bonds.
class ConnectedCluster:
    def __init__(self, bonds):
        self.bonds = bonds
        self.cc = self.find_connected_components()

    def find_connected_components(self):
        '''The function `find_connected_components` finds the connected components in a graph represented by a
        list of bonds.
        
        Returns
        -------
            The code is returning the variable "connected_components", which is a list of lists representing
        the connected components in the graph.
        
        '''
        connections = {}

        for bond in self.bonds:
            for item in bond:
                if item not in connections:
                    connections[item] = {item}

        for bond in self.bonds:
            i, j = bond
            if i != j:
                connections[i].add(j)
                connections[j].add(i)

        visited = set()
        connected_components = []

        for node in connections:
            if node not in visited:
                component = self.dfs(node, connections, visited)
                connected_components.append(list(component))

        return connected_components

    def dfs(self, node, connections, visited):
        '''The above function implements a depth-first search algorithm to find all connected nodes in a graph.
        
        Parameters
        ----------
        node
            The "node" parameter represents the starting node for the depth-first search algorithm. It is the
        node from which the search will begin exploring the connected components of the graph.
        connections
            The "connections" parameter is a dictionary that represents the connections between nodes. Each key
        in the dictionary represents a node, and the corresponding value is a list of nodes that are
        connected to the key node.
        visited
            The "visited" parameter is a set that keeps track of the nodes that have already been visited
        during the depth-first search traversal.
        
        Returns
        -------
            a set of nodes that are part of the connected component starting from the given node.
        
        '''
        stack = [node]
        component = set()

        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                component.add(current_node)
                visited.add(current_node)
                stack.extend(connections[current_node])

        return component


class FetchCluster:
    def __init__(self, universe, frame, num_atoms, thresh1 = 100, thresh2 = 25, verbose = False):
        '''The function initializes various variables and performs clustering on a given dataset, extracting
        the largest cluster.
        
        Parameters
        ----------
        universe
            The `universe` parameter represents the molecular dynamics simulation system or trajectory that you
        are working with. It could be a `MDAnalysis.Universe` object or any other object that contains the
        necessary information about the system.
        frame
            The `frame` parameter represents the frame number in the `universe` object. It is used to select a
        specific frame from the trajectory data.
        num_atoms
            The `num_atoms` parameter represents the total number of atoms in the system. It is used to specify
        the range of atoms to select for clustering and other calculations.
        thresh1, optional
            The `thresh1` parameter is a threshold value used in the clustering algorithm. It determines the
        maximum distance between two atoms for them to be considered part of the same cluster. If the
        distance between two atoms is greater than `thresh1`, they will be assigned to different clusters.
        thresh2, optional
            The `thresh2` parameter is a threshold value used in the clustering algorithm. It is used to
        determine the minimum distance between atoms in order to consider them as part of the same cluster.
        If the distance between two atoms is less than `thresh2`, they are considered to be in the same
        cluster
        verbose, optional
            The `verbose` parameter is a boolean flag that determines whether or not to print additional
        information during the execution of the code. If `verbose` is set to `True`, it will print the
        number of clusters and the number of scattered molecules. If `verbose` is set to `False`, it
        
        '''
        # sourcery skip: low-code-quality
        self.universe = universe
        self.frame = frame
        ### TODO make framewise something
        universe.trajectory[frame]
        self.dimensions = universe.dimensions
        self.num_atoms = num_atoms
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        mindists = self.mindistMaker(self.universe, frame, num_atoms)
        self.pos = self.universe.select_atoms(f"resid 1-{self.num_atoms}").positions
        self.resids = self.universe.select_atoms(f'resid 1-{self.num_atoms}').resids
        # First clustering..
        THC = Hierarchical_Clastering()
        THC.clusters(mindists, self.thresh1, no_plot=False)
        miniclusts, sliced_minD = self.getSliceFromcluster(np.arange(self.num_atoms), THC.labels, min_dist = mindists)
        # Initiator 
        one_mol_cluster = []
        large_cluster = []
        large_cluster_distances = []

        for sg, clust in zip(sliced_minD, miniclusts):
            if (
                len(clust) != 1
                and len(clust) == 2
                and sg[0, 1] > 3.5
                or len(clust) == 1
            ):
                one_mol_cluster.extend(clust)
            elif len(clust) == 2:
                large_cluster.append(clust)
                large_cluster_distances.append(sg)
            elif len(clust) > 2:
                #large_cluster.append(clust)
                new_d = sg.copy()
                orphan_indx = []

                for i in range(len(sg)):
                    if (np.where(sg[i] < 3.5)[0].shape[0]) == 1:
                        #print(clust[i])
                        orphan_indx.append(i)
                        one_mol_cluster.extend([clust[i]])
                new_d = np.delete(new_d, orphan_indx, axis = 0)
                new_d = np.delete(new_d, orphan_indx, axis = 1)
                if len(new_d) > 1:
                    if len(new_d) == 2:
                        large_cluster.append(clust[~np.isin(clust, clust[orphan_indx])])
                        large_cluster_distances.append(new_d)
                    else:
                        ## Check for True clusters

                        oc, bc, bcd = self.break_further(new_d, clust[~np.isin(clust, clust[orphan_indx])])
                        one_mol_cluster.extend(oc)
                        if (len(bc)):
                            for c in bc :
                                large_cluster.append(c)
                                large_cluster_distances.append(bcd)


        self.bondDetector(large_cluster, self.resids, self.pos, self.dimensions)
        CC = ConnectedCluster(np.array(self.bonds))
        self.connected_large = CC.cc
        #print(CC.cc)
        self.excluded_from_large_cluster = self.flatten_comprehension(CC.cc)
        self.remaining_in_large_cluster = np.arange(len(large_cluster))[~np.isin(np.arange(len(large_cluster)), np.array(self.excluded_from_large_cluster))]
        #print(remaining_in_large_cluster)
        all_cluster = []
        self.large_cluster = large_cluster
        for j in range(len(self.connected_large)):
            temp_list = []
            for i in self.connected_large[j]:
                temp_list.extend(self.large_cluster[i])
            all_cluster.append(temp_list)
        all_cluster.extend(
            self.large_cluster[k] for k in self.remaining_in_large_cluster
        )
        self.all_cluster = all_cluster
        ln = self.flatten_comprehension(self.all_cluster)
        if verbose:
            print(f"Number of clusters : {len(self.all_cluster)}")
            print("Number of scattered molecules : ", self.num_atoms - len(ln))
        self.total_number_of_clusters = len(self.all_cluster) + self.num_atoms - len(ln)
        self.size_of_clusters = np.array(list(map(lambda x : len(x) , self.all_cluster)))
        self.max_cluster_size = 1
        if len(self.size_of_clusters):
            self.max_cluster_id = self.all_cluster[np.argmax(self.size_of_clusters)]
            self.max_cluster_size = len(self.max_cluster_id)
            if len(self.max_cluster_id):
                self.extract_max_cluster(self.max_cluster_id)
    
    def mindistMaker(self, u, frame, num_atoms):
        '''The `mindistMaker` function calculates the minimum distance between pairs of atoms in a given frame
        of a trajectory.
        
        Parameters
        ----------
        u
            The parameter "u" is likely an object representing a molecular dynamics simulation trajectory. It
        is used to access the coordinates and other properties of the atoms in the system at a given frame.
        frame
            The `frame` parameter represents the frame number of the trajectory. It is used to set the current
        frame of the `Universe` object `u` to the specified frame number using `u.trajectory[frame]`. This
        allows you to calculate the minimum distance between atoms in that specific frame.
        num_atoms
            The parameter `num_atoms` represents the total number of atoms in the system.
        
        Returns
        -------
            a 2D numpy array called `min_dist`. This array contains the minimum distances between pairs of
        atoms in the system.
        
        '''
        u.trajectory[frame]
        sel_dict = {
            i: u.select_atoms(f"resid {i+1} and not name CL2", updating=True) for i in range(num_atoms)
        }
        min_dist = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms-1):
            for j in range(i+1, num_atoms):
                d = np.min(distance_array(sel_dict[i].positions, sel_dict[j].positions, box = u.dimensions))
                min_dist[i, j] = d
                min_dist[j, i] = d
        
        return min_dist
    

    def getSliceFromcluster(self, cluster, HCL, min_dist):
        '''The function `getSliceFromcluster` takes in a cluster, a hierarchical clustering object, and a
        minimum distance matrix, and returns mini clusters and sliced distances.
        
        Parameters
        ----------
        cluster
            The `cluster` parameter is a numpy array that represents the clustering of data points. Each
        element in the array corresponds to the cluster assignment of a data point.
        HCL
            HCL is a numpy array that represents the hierarchical clustering labels for each data point. It
        is used to group similar data points together based on their distances.
        min_dist
            The `min_dist` parameter is a matrix that represents the minimum distance between each pair of
        points in the dataset. It is used to calculate the distances between points within each cluster.
        
        Returns
        -------
            two lists: mini_clusters and sliced_distances. mini_clusters contains the clusters obtained
        from the input cluster based on the hierarchical clustering labels (HCL). sliced_distances
        contains the pairwise distances between the elements within each cluster.
        
        '''
        unique_clust = np.unique(HCL)
        num_clusters = len(unique_clust)
        mini_clusters = []
        sliced_distances = []
        for i in range(num_clusters):
            myclust = cluster[HCL==i]
            n = myclust.shape[0]
            sliced = np.zeros((n, n))
            for i in range(n-1):
                for j in range(i+1, n):
                    sliced[i, j] = min_dist[myclust[i], myclust[j]]
                    sliced[j, i] = min_dist[myclust[j], myclust[i]]
                sliced[i, i] = 0
            sliced[n-1, n-1] = 0
            mini_clusters.append(myclust)
            sliced_distances.append(sliced)
        return mini_clusters, sliced_distances
    

    check_near = lambda self, buf : np.where(np.array([np.where(buf[i] < 3.5)[0].shape[0] for i in range(len(buf))]) > 2)[0].shape[0]
    
    def break_further(self, sg2, br2):
        '''The function "break_further" takes in two arrays, "sg2" and "br2", and performs hierarchical
        clustering on "sg2" to identify clusters. It then checks each cluster to determine if it should be
        considered a broken cluster or not, based on certain conditions. The function returns three arrays:
        "other_clusters" contains clusters that do not meet the conditions to be considered broken clusters,
        "broken_clusters" contains clusters that meet the conditions to be considered broken clusters, and
        "broken_clusters_distances" contains the distances between the points in the broken clusters.
        
        Parameters
        ----------
        sg2
            The parameter `sg2` is a list or array containing the data points or samples that need to be
        clustered.
        br2
            The parameter `br2` is a list of elements that represent clusters.
        
        Returns
        -------
            three values: `other_clusters`, `broken_clusters`, and `broken_clusters_distances`.
        
        '''
        HC = Hierarchical_Clastering()
        HC.clusters(sg2, self.thresh2, no_plot=False)
        miniclusts1, sliced_minD1 = self.getSliceFromcluster(np.arange(len(sg2)), HC.labels, min_dist = sg2)
        #print(miniclusts1)
        broken_clusters = []
        other_clusters = []
        broken_clusters_distances = []
        for i, dm in enumerate(sliced_minD1):
            if len(dm) > 1:
                cn = self.check_near(dm)
                orphan = len(dm) - np.where(np.array([np.where(dm[i]< 3.5)[0].shape[0] for i in range(len(dm))]) > 2)[0].shape[0]
                #print(orphan)
                if orphan == 0 and len(dm) < 4 and cn > 1:
                    broken_clusters.append(np.array(br2)[miniclusts1[i]])
                    broken_clusters_distances.append(dm)
                elif len(dm) == 2 and cn == 0:
                    other_clusters.extend(np.array(br2)[miniclusts1[i]])
                else:
                    broken_clusters.append(np.array(br2)[miniclusts1[i]])
                    broken_clusters_distances.append(dm)

            else:
                other_clusters.extend(np.array(br2)[miniclusts1[i]])
        return other_clusters, broken_clusters, broken_clusters_distances
    
    def connection_cluster(self, cluster_indx1, cluster_indx2, all_resids, all_pos, dimensions):
        '''The function `connection_cluster` calculates the total number of connections between two clusters of
        molecules based on their positions and dimensions.
        
        Parameters
        ----------
        cluster_indx1
            cluster_indx1 is a list of indices representing the first cluster of molecules.
        cluster_indx2
            The parameter `cluster_indx2` is a list of indices representing the second cluster.
        all_resids
            The parameter `all_resids` is a numpy array that contains the residue indices for all atoms in the
        system.
        all_pos
            The variable `all_pos` is a numpy array that contains the positions of all atoms in the system.
        Each row of `all_pos` represents the position of an atom, and the columns represent the x, y, and z
        coordinates of the atom.
        dimensions
            The dimensions parameter represents the number of dimensions in the coordinate system. It is used
        to calculate the distance between two points in the distance_array function.
        
        Returns
        -------
            the total number of connections between molecules in cluster_indx1 and cluster_indx2 that have a
        distance less than 3.5 units.
        
        '''
        total_connection = 0
        for i in cluster_indx1:
            mol1 = all_pos[np.where(all_resids == i+1)[0]]
            for j in cluster_indx2:
                mol2 = all_pos[np.where(all_resids == j+1)[0]]
                d = distance_array(mol1, mol2, dimensions)
                connections = np.where(d < 3.5)[0].shape[0]
                if connections > 2:
                    total_connection += 1
        return total_connection
    
    def bondDetector(self, large_cluster, all_resids, all_pos, dimensions):
        '''The `bondDetector` function calculates the connectivity matrix between atoms in a large cluster and
        identifies bonds based on a distance threshold.
        
        Parameters
        ----------
        large_cluster
            The `large_cluster` parameter is a list containing the indices of atoms in a cluster.
        all_resids
            The `all_resids` parameter is a list that contains the residue numbers of all atoms in the system.
        all_pos
            The parameter "all_pos" is a list of positions of all atoms in the system. Each position is
        represented as a tuple of three coordinates (x, y, z).
        dimensions
            The "dimensions" parameter is the dimensions of the system in which the large cluster exists. It
        specifies the size of the system in terms of its x, y, and z dimensions.
        
        '''
        conn_mat = np.zeros((len(large_cluster), len(large_cluster)))
        bonds = []
        for i in range(len(large_cluster)-1):
            for j in range(i+1, len(large_cluster)):
                
                d = self.connection_cluster(large_cluster[i], large_cluster[j], all_resids, all_pos, dimensions)

                conn_mat[i, j] = d
                conn_mat[j, i] = d
                ## TODO Another criteria 
                if d > 1:
                    bonds.append([i, j])
        self.bonds = bonds

    def flatten_comprehension(self, matrix):
        '''The function `flatten_comprehension` takes a matrix as input and returns a flattened version of the
        matrix using list comprehension.
        
        Parameters
        ----------
        matrix
            A 2-dimensional list or matrix.
        
        Returns
        -------
            a flattened version of the input matrix using a list comprehension.
        
        '''
        return [item for row in matrix for item in row]
    
    def extract_max_cluster(self, indx_list):
        '''The function extracts the positions of atoms in a cluster specified by a list of indices.
        
        Parameters
        ----------
        indx_list
            The `indx_list` parameter is a list of indices. It is used to select specific residues from the
        `universe` object. The residues are selected based on their indices in the `indx_list`. The
        selected residues are then used to extract the positions of atoms in the `max_cluster` attribute
        
        '''
        select_str = "".join(f"resid {indx + 1} or " for indx in indx_list[:-1])
        select_str += f"resid {indx_list[-1] + 1}"
        u = self.universe
        u.trajectory[self.frame]
        self.max_cluster = u.select_atoms(select_str).positions

    def create_visulization(self):
        '''The function `create_visulization` creates a visualization of a molecular system using the
        MDAnalysis library in Python.
        
        '''
        colorlist = ["tv_green", "marine", "purpleblue", "tv_yellow", "lightmagenta", "violet", "greencyan", 
                    "deepteal", "tv_orange", "brown", "iron", "lead", "mercury", "antimony", "salmon", "olive", "br5", "black", "limon", "raspberry", "skyblue"]
        u = self.universe
        u.trajectory[self.frame]
        ag = self.universe.select_atoms(f"resid 1-{self.num_atoms}")     
        
        with mda.Writer("vis.pdb", multiframe=True) as W:
            W.write(ag)
       
        f = open("vis.pml", "w")
        f.write("load vis.pdb \n")
        for i, element in enumerate(self.all_cluster):
            selection = f"select clust{i}, "
            for indx in element[:-1]:
                selection += f"resid {indx +1} or "
            selection +=  f"resid {element[-1] +1}"
            f.write(selection + "\n")
        for i in range(len(self.all_cluster)):
            if i < 20:
                f.write(f"color {colorlist[i]}, clust{i} \n")
        f.write("hide everything \n")
        f.write("show sticks \n")
        f.write("hide everything, hydrogens \n")
        f.write("show cell \n")

