# Functions to Calculate Aromatic Stacking Interactions
from __future__ import print_function, division
from mdtraj.geometry import _geometry
from mdtraj.utils import ensure_type
from mdtraj.geometry import compute_distances, compute_angles

import numpy as np
import pandas as pd
import os
from os.path import join
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mdtraj as md
import seaborn as sns
import scipy as sp
from numpy import log2, zeros, mean, var, sum, loadtxt, arange, \
    array, cumsum, dot, transpose, diagonal, floor
import sklearn.utils.validation as suv
from scipy.interpolate import make_interp_spline
import pyblock
from numpy.linalg import inv, lstsq
import math


def find_plane_normal(points):

    N = points.shape[0]
    A = np.concatenate((points[:, 0:2], np.ones((N, 1))), axis=1)
    B = points[:, 2]
    out = lstsq(A, B, rcond=-1)
    na_c, nb_c, d_c = out[0]
    if d_c != 0.0:
        cu = 1./d_c
        bu = -nb_c*cu
        au = -na_c*cu
    else:
        cu = 1.0
        bu = -nb_c
        au = -na_c
    normal = np.asarray([au, bu, cu])
    normal /= math.sqrt(dot(normal, normal))
    return normal


def find_plane_normal2(positions):
    # Alternate approach used to check sign - could the sign check cause descrepency with desres?
    # Use Ligand IDs 312, 308 and 309 to check direction
    # [304 305 306 307 308 309 310 311 312 313]
    v1 = positions[0]-positions[1]
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = positions[2]-positions[1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal = np.cross(v1, v2)
    return normal


def find_plane_normal2_assign_atomid(positions, id1, id2, id3):
    # Alternate approach used to check sign - could the sign check cause descrepency with desres?
    v1 = positions[id1]-positions[id2]
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = positions[id3]-positions[id1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal = np.cross(v1, v2)
    return normal


def get_ring_center_normal_assign_atomid(positions, id1, id2, id3):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2_assign_atomid(positions, id1, id2, id3)
    # check direction of normal using dot product convention
    comp = np.dot(normal, normal2)
    if comp < 0:
        normal = -normal
    return center, normal


def get_ring_center_normal_(positions):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2(positions)
    # check direction of normal using dot product convention
    comp = np.dot(normal, normal2)
    if comp < 0:
        normal = -normal
    return center, normal


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1))*np.sqrt(np.dot(v2, v2))))


def get_ring_center_normal_trj_assign_atomid(position_array, id1, id2, id3):
    length = len(position_array)
    centers = np.zeros((length, 3))
    normals = np.zeros((length, 3))
    centers_normals = np.zeros((length, 2, 3))
    for i in range(0, len(position_array)):
        center, normal = get_ring_center_normal_assign_atomid(
            position_array[i], id1, id2, id3)
        centers_normals[i][0] = center
        centers_normals[i][1] = normal
    return centers_normals

# MDtraj Functions to Calculate Hydrogen Bonds with custom selections of donors and acceptors

def _get_bond_triplets_print(topology, lig_donors, exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        # print("get_donors e0 e1:",e0,e1)
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
                 if set((one.element.symbol, two.element.symbol)) == elems]
        # Filter non-participating atoms
        atoms = [atom for atom in atoms
                 if can_participate(atom[0]) and can_participate(atom[1])]
        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break  # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')

    nh_donors = get_donors('N', 'H')
    oh_donors = get_donors('O', 'H')
    sh_donors = get_donors('S', 'H')
    # ADD IN ADDITIONAL SPECIFIED LIGAND DONORS
    xh_donors = np.array(nh_donors + oh_donors + sh_donors+lig_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N', 'S'))
    acceptors = [a.index for a in topology.atoms
                 if a.element.symbol in acceptor_elements and can_participate(a)]
    print("acceptors")
    for i in acceptors:
        print(top.atom(i))
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]


def _get_bond_triplets(topology, lig_donors, exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
                 if set((one.element.symbol, two.element.symbol)) == elems]
        # Filter non-participating atoms
        atoms = [atom for atom in atoms
                 if can_participate(atom[0]) and can_participate(atom[1])]
        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break  # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')

    nh_donors = get_donors('N', 'H')
    oh_donors = get_donors('O', 'H')
    sh_donors = get_donors('S', 'H')
    xh_donors = np.array(nh_donors + oh_donors + sh_donors+lig_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N', 'S'))
    acceptors = [a.index for a in topology.atoms
                 if a.element.symbol in acceptor_elements and can_participate(a)]
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]


def _compute_bounded_geometry(traj, triplets, distance_cutoff, distance_indices,
                              angle_indices, freq=0.0, periodic=True):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = md.compute_distances(
        traj, triplets[:, distance_indices], periodic=periodic)

    # Now we discover which triplets meet the distance cutoff often enough
    prevalence = np.mean(distances < distance_cutoff, axis=0)
    mask = prevalence > freq

    # Update data structures to ignore anything that isn't possible anymore
    triplets = triplets.compress(mask, axis=0)
    distances = distances.compress(mask, axis=1)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(md.compute_distances(traj, triplets[:, abc_pair],
                                                      periodic=periodic))

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines)  # avoid NaN error
    angles = np.arccos(cosines)
    return mask, distances, angles


def baker_hubbard2(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False,
                   distance_cutoff=0.35, angle_cutoff=150, lig_donor_index=[]):

    angle_cutoff = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets

    # ADD IN LIGAND HBOND DONORS
    add_donors = lig_donor_index

    bond_triplets = _get_bond_triplets(traj.topology,
                                       exclude_water=exclude_water, lig_donors=add_donors, sidechain_only=sidechain_only)

    mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
                                                        distance_cutoff, [1, 2], [0, 1, 2], freq=freq, periodic=periodic)

    # Find triplets that meet the criteria
    presence = np.logical_and(
        distances < distance_cutoff, angles > angle_cutoff)
    mask[mask] = np.mean(presence, axis=0) > freq
    return bond_triplets.compress(mask, axis=0)


def print_donors_acceptors(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False,
                           distance_cutoff=0.35, angle_cutoff=150, lig_donor_index=[]):

    angle_cutoff = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets

    # ADD IN LIGAND HBOND DONORS
    # add_donors=[[296,318],[296,331]]
    # Manually tell it where to find proton donors on ligand
    # LIG58-O5 LIG58-H24
    # LIG58-O1 LIG58-H12
    # LIG58-N LIG58-H15
    # add_donors=[[768,796],[750,784],[752,787]]
    add_donors = lig_donor_index

    bond_triplets_print = _get_bond_triplets_print(traj.topology,
                                                   exclude_water=exclude_water, lig_donors=add_donors, sidechain_only=sidechain_only)

    # mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
    #    distance_cutoff, [1, 2], [0, 1, 2], freq=freq, periodic=periodic)

    # Find triplets that meet the criteria
    # presence = np.logical_and(distances < distance_cutoff, angles > angle_cutoff)
    # mask[mask] = np.mean(presence, axis=0) > freq
    return

def add_contact_pair(pairs, a1, a2, a1_id, a2_id, prot_res, contact_prob):
    if prot_res not in pairs:
        pairs[prot_res] = {}
    if a2 not in pairs[prot_res]:
        pairs[prot_res][a2] = {}
    if a1_id not in pairs[prot_res][a2]:
        pairs[prot_res][a2][a1_id] = contact_prob

def normvector_connect(point1, point2):
    vec = point1-point2
    vec = vec/np.sqrt(np.dot(vec, vec))
    return vec


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1))*np.sqrt(np.dot(v2, v2))))

def add_hbond_pair(donor, acceptor, hbond_pairs, donor_res):
    if donor_res not in hbond_pairs:
        hbond_pairs[donor_res] = {}
    if donor not in hbond_pairs[donor_res]:
        hbond_pairs[donor_res][donor] = {}
    if acceptor not in hbond_pairs[donor_res][donor]:
        hbond_pairs[donor_res][donor][acceptor] = 0
    hbond_pairs[donor_res][donor][acceptor] += 1

def calc_contact(trj, ligand_residue_index):
    residues = 58
    contact_pairs = np.zeros((residues, 2))

    for i in range(0, residues):
        contact_pairs[i] = [i, ligand_residue_index]
    contact = md.compute_contacts(trj, contact_pairs, scheme='closest-heavy')
    contacts = np.asarray(contact[0]).astype(float)
    cutoff = 0.6
    contact_matrix = np.where(contacts < cutoff, 1, 0)

    return contact_matrix

def calc_hphob(trj, ligand_residue_index):
    top = trj.topology
    n_frames = trj.n_frames
    residues = 24
    ligand_hphob = top.select("residue %s and element C" % str(ligand_residue_index))
    protein_hphob = top.select("residue 1 to 24 and element C")
    
    hphob_pairs = []
    for i in ligand_hphob:
        for j in protein_hphob:
            hphob_pairs.append([i, j])


    contact = md.compute_distances(trj, hphob_pairs)
    contacts = np.asarray(contact).astype(float)
    cutoff = 0.4
    contact_frames = np.where(contacts < cutoff, 1, 0)
    contact_prob_hphob = np.sum(contact_frames, axis=0)/trj.n_frames

    Hphob_res_contacts = np.zeros((n_frames, residues))
    for frame in range(n_frames):
        if np.sum(contact_frames[frame]) > 0:
            contact_pairs = np.where(contact_frames[frame] == 1)
            for j in contact_pairs[0]:
                residue = top.atom(hphob_pairs[j][1]).residue.resSeq
                Hphob_res_contacts[frame][residue] = 1

    return Hphob_res_contacts

def calc_hphob_dimer(trj, ligand_residue_index):
    top = trj.topology
    n_frames = trj.n_frames
    residues = 24
    ligand_hphob = top.select("residue %s and element C" % str(ligand_residue_index))
    protein_hphob = top.select("residue 1 to 24 and element C")
    
    hphob_pairs = []
    for i in ligand_hphob:
        for j in protein_hphob:
            hphob_pairs.append([i, j])


    contact = md.compute_distances(trj, hphob_pairs)
    contacts = np.asarray(contact).astype(float)
    cutoff = 0.4
    contact_frames = np.where(contacts < cutoff, 1, 0)
    contact_prob_hphob = np.sum(contact_frames, axis=0)/trj.n_frames

    Hphob_res_contacts = np.zeros((n_frames, residues))
    for frame in range(n_frames):
        if np.sum(contact_frames[frame]) > 0:
            contact_pairs = np.where(contact_frames[frame] == 1)
            for j in contact_pairs[0]:
                residue = top.atom(hphob_pairs[j][1]).residue.resSeq
                Hphob_res_contacts[frame][residue-1] = 1

    return Hphob_res_contacts

def calc_aromatic(trj, ligand_rings):
    n_frames = trj.n_frames
    top = trj.topology
    n_rings = len(ligand_rings)
    residues = 58
    residue_offset = 390
    
    ligand_ring_params = []
    for i in range(0, n_rings):
        ring = np.array(ligand_rings[i])
        positions = trj.xyz[:, ring, :]
        ligand_centers_normals = get_ring_center_normal_trj_assign_atomid(
            positions, 0, 1, 2)
        ligand_ring_params.append(ligand_centers_normals)

    prot_rings = []
    aro_residues = []
    prot_ring_name = []
    prot_ring_index = []

    aro_select = top.select("resname TYR PHE HIS TRP and name CA")
    for i in aro_select:
        atom = top.atom(i)
        resname = atom.residue.name
        if resname == "TYR":
            ring = top.select(
                "resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)
        if resname == "TRP":
            ring = top.select(
                "resid %s and name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2" % atom.residue.index)
        if resname == "HIS":
            ring = top.select("resid %s and name CG ND1 CE1 NE2 CD2" %
                            atom.residue.index)
        if resname == "PHE":
            ring = top.select(
                "resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)
        prot_rings.append(ring)
        prot_ring_name.append(atom.residue)
        prot_ring_index.append(atom.residue.index+residue_offset)

    prot_ring_params = []
    for i in range(0, len(prot_rings)):
        ring = np.array(prot_rings[i])
        positions = trj.xyz[:, ring, :]
        ring_centers_normals = get_ring_center_normal_trj_assign_atomid(
            positions, 0, 1, 2)
        prot_ring_params.append(ring_centers_normals)
        
    frames = n_frames
    sidechains = len(prot_rings)
    ligrings = len(ligand_rings)
    Ringstacked_old = {}
    Ringstacked = {}
    Quadrants = {}
    Stackparams = {}
    Aro_Contacts = {}
    Pstack = {}
    Tstack = {}
        
    for l in range(0, ligrings):
        name = "Lig_ring.%s" % l
        # print(name)
        Stackparams[name] = {}
        Pstack[name] = {}
        Tstack[name] = {}
        Aro_Contacts[name] = {}
        alphas = np.zeros(shape=(frames, sidechains))
        betas = np.zeros(shape=(frames, sidechains))
        dists = np.zeros(shape=(frames, sidechains))
        thetas = np.zeros(shape=(frames, sidechains))
        phis = np.zeros(shape=(frames, sidechains))
        pstacked_old = np.zeros(shape=(frames, sidechains))
        pstacked = np.zeros(shape=(frames, sidechains))
        tstacked = np.zeros(shape=(frames, sidechains))
        stacked = np.zeros(shape=(frames, sidechains))
        aro_contacts = np.zeros(shape=(frames, sidechains))

        for i in range(0, frames):
            ligcenter = ligand_ring_params[l][i][0]
            lignormal = ligand_ring_params[l][i][1]
            for j in range(0, sidechains):
                protcenter = prot_ring_params[j][i][0]
                protnormal = prot_ring_params[j][i][1]
                dists[i, j] = np.linalg.norm(ligcenter-protcenter)
                connect = normvector_connect(protcenter, ligcenter)
                # alpha is the same as phi in gervasio/Procacci definition
                alphas[i, j] = np.rad2deg(angle(connect, protnormal))
                betas[i, j] = np.rad2deg(angle(connect, lignormal))
                theta = np.rad2deg(angle(protnormal, lignormal))
                thetas[i, j] = np.abs(theta)-2*(np.abs(theta)
                                                > 90.0)*(np.abs(theta)-90.0)
                phi = np.rad2deg(angle(protnormal, connect))
                phis[i, j] = np.abs(phi)-2*(np.abs(phi) > 90.0)*(np.abs(phi)-90.0)

        for j in range(0, sidechains):
            name2 = prot_ring_index[j]
            res2 = prot_ring_name[j]
            Ringstack = np.column_stack(
                (dists[:, j], alphas[:, j], betas[:, j], thetas[:, j], phis[:, j]))
            stack_distance_cutoff = 0.65
            r = np.where(dists[:, j] <= stack_distance_cutoff)[0]
            aro_contacts[:, j][r] = 1

            # New Definitions
            # p-stack: r < 6.5 Å, θ < 60° and ϕ < 60°.
            # t-stack: r < 7.5 Å, 75° < θ < 90° and ϕ < 60°.
            p_stack_distance_cutoff = 0.65
            t_stack_distance_cutoff = 0.75
            r_pstrict = np.where(dists[:, j] <= p_stack_distance_cutoff)[0]
            r_tstrict = np.where(dists[:, j] <= t_stack_distance_cutoff)[0]

            e = np.where(thetas[:, j] <= 45)
            f = np.where(phis[:, j] <= 60)
            g = np.where(thetas[:, j] >= 75)

            pnew = np.intersect1d(np.intersect1d(e, f), r_pstrict)
            tnew = np.intersect1d(np.intersect1d(g, f), r_tstrict)
            pstacked[:, j][pnew] = 1
            tstacked[:, j][tnew] = 1
            stacked[:, j][pnew] = 1
            stacked[:, j][tnew] = 1
            total_stacked = len(pnew)+len(tnew)
            Stackparams[name][name2] = Ringstack
        Pstack[name] = pstacked
        Tstack[name] = tstacked
        Aro_Contacts[name] = aro_contacts
        Ringstacked[name] = stacked

    residue_number = range(residue_offset, residue_offset+residues)
    aro_res_index = np.array(prot_ring_index)-residue_offset
    aromatic_stacking_contacts = np.zeros((n_frames, residues))
    aromatic_contacts = np.zeros((n_frames, residues))

    for i in range(0, len(aro_res_index)):
        if ligrings == 1:
            aromatic_stacking_contacts[:, aro_res_index[i]
                                       ] += Ringstacked['Lig_ring.0'][:, i]
        else:
            aromatic_stacking_contacts[:, aro_res_index[i]
                                    ] += Ringstacked['Lig_ring.0'][:, i]
            aromatic_stacking_contacts[:, aro_res_index[i]
                                    ] += Ringstacked['Lig_ring.1'][:, i]

    return aromatic_stacking_contacts

def calc_hbond(trj, ligand, lig_hbond_donors):
    n_frames = trj.n_frames
    top = trj.topology
    residues = 58
    residue_offset = 390
    residue_number = range(residue_offset, residue_offset+residues)
    # Select Ligand Residues
    ligand = top.select("residue %s" % str(ligand))
    # Select Protein Residues
    protein = top.select("residue 0 to 57")


    HBond_PD = np.zeros((n_frames, residues))
    HBond_LD = np.zeros((n_frames, residues))
    Hbond_pairs_PD = {}
    Hbond_pairs_LD = {}

    for frame in range(n_frames):
        hbonds = baker_hubbard2(trj[frame], angle_cutoff=150,
                                distance_cutoff=0.35, lig_donor_index=lig_hbond_donors)
        for hbond in hbonds:
            if ((hbond[0] in protein) and (hbond[2] in ligand)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc = top.atom(hbond[2])
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_PD[frame][donor_res] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_PD, donor_res)
            if ((hbond[0] in ligand) and (hbond[2] in protein)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc_id = hbond[2]
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_LD[frame][acc_res] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_LD, acc_res)

    HB_Total = HBond_PD+HBond_LD
    
    return HB_Total

def calc_hbond_dimer_m2d(trj, ligand, lig_hbond_donors):
    n_frames = trj.n_frames
    top = trj.topology
    residues = 24
    residue_offset = 390
    residue_number = range(residue_offset, residue_offset+residues)
    # Select Ligand Residues
    ligand = top.select("residue %s" % str(ligand))
    # Select Protein Residues
    protein = top.select("residue 1 to 24")


    HBond_PD = np.zeros((n_frames, residues))
    HBond_LD = np.zeros((n_frames, residues))
    Hbond_pairs_PD = {}
    Hbond_pairs_LD = {}

    for frame in range(n_frames):
        hbonds = baker_hubbard2(trj[frame], angle_cutoff=150,
                                distance_cutoff=0.35, lig_donor_index=lig_hbond_donors)
        for hbond in hbonds:
            if ((hbond[0] in protein) and (hbond[2] in ligand)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc = top.atom(hbond[2])
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_PD[frame][donor_res-1] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_PD, donor_res)
            if ((hbond[0] in ligand) and (hbond[2] in protein)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc_id = hbond[2]
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_LD[frame][acc_res-1] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_LD, acc_res)

    HB_Total = HBond_PD+HBond_LD
    
    return HB_Total

def calc_hbond_dimer_m1d(trj, ligand, lig_hbond_donors):
    n_frames = trj.n_frames
    top = trj.topology
    residues = 24
    residue_offset = 390
    residue_number = range(residue_offset, residue_offset+residues)
    # Select Ligand Residues
    ligand = top.select("residue %s" % str(ligand))
    # Select Protein Residues
    protein = top.select("residue 27 to 51")

    HBond_PD = np.zeros((n_frames, residues))
    HBond_LD = np.zeros((n_frames, residues))
    Hbond_pairs_PD = {}
    Hbond_pairs_LD = {}

    for frame in range(n_frames):
        hbonds = baker_hubbard2(trj[frame], angle_cutoff=150,
                                distance_cutoff=0.35, lig_donor_index=lig_hbond_donors)
        for hbond in hbonds:
            if ((hbond[0] in protein) and (hbond[2] in ligand)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc = top.atom(hbond[2])
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_PD[frame][donor_res-28] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_PD, donor_res)
            if ((hbond[0] in ligand) and (hbond[2] in protein)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc_id = hbond[2]
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_LD[frame][acc_res-28] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_LD, acc_res)

    HB_Total = HBond_PD+HBond_LD
    
    return HB_Total

def dssp_convert(dssp):
    dsspH = np.copy(dssp)
    dsspE = np.copy(dssp)
    dsspH[dsspH == 'H'] = 1
    dsspH[dsspH == 'E'] = 0
    dsspH[dsspH == 'C'] = 0
    dsspH[dsspH == 'NA'] = 0
    dsspH = dsspH.astype(int)
    TotalH = np.sum(dsspH, axis=1)
    SE_H = np.zeros((len(dssp[0]), 2))

    for i in range(0, len(dssp[0])):
        data = dsspH[:, i].astype(float)
        if(np.mean(data) > 0):
            SE_H[i] = [np.mean(data), (block(data))**.5]

    dsspE[dsspE == 'H'] = 0
    dsspE[dsspE == 'E'] = 1
    dsspE[dsspE == 'C'] = 0
    dsspE[dsspE == 'NA'] = 0
    dsspE = dsspE.astype(int)
    TotalE = np.sum(dsspE, axis=1)
    Eprop = np.sum(dsspE, axis=0).astype(float)/len(dsspE)
    SE_E = np.zeros((len(dssp[0]), 2))

    for i in range(0, len(dssp[0])):
        data = dsspE[:, i].astype(float)
        if(np.mean(data) > 0):
            SE_E[i] = [np.mean(data), (block(data))**.5]
    return SE_H, SE_E


# block function from pyblock package -  https://github.com/jsspencer/pyblock
def block(x):
    # preliminaries
    d = log2(len(x))
    if (d - floor(d) != 0):
        #    print("Warning: Data size = %g, is not a power of 2." % floor(2**d))
        #    print("Truncating data to %g." % 2**floor(d) )
        x = x[:2**int(floor(d))]
    d = int(floor(d))
    n = 2**d
    s, gamma = zeros(d), zeros(d)
    mu = mean(x)
    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in arange(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum((x[0:(n-1)]-mu)*(x[1:n]-mu))
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum(((gamma/s)**2*2**arange(1, d+1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
               16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
               24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
               31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
               38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
               45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0, d):
        if(M[k] < q[k]):
            break
    # if (k >= d-1):
    #     print("Warning: Use more data")

    return (s[k]/2**(d-k))


def Kd_calc(bound, conc):
    return((1-bound)*conc/bound)


def get_blockerrors(Data, bound_frac):
    n_data = len(Data[0])
    block_errors = []
    ave = []
    for i in range(0, n_data):
        data = Data[:, i]
        average = np.average(data)
        be = block(data)**.5
        ave.append(np.average(data))
        block_errors.append(be)
    ave_bf = np.asarray(ave)/bound_frac
    be_bf = np.asarray(block_errors)/bound_frac

    return ave_bf, be_bf


def get_blockerrors_pyblock(Data, bound_frac):
    n_data = len(Data[0])
    block_errors = []
    ave = []
    for i in range(0, n_data):
        data = Data[:, i]
        average = np.average(data)
        if (average != 0) and (average != 1):
            reblock_data = pyblock.blocking.reblock(data)
            opt = pyblock.blocking.find_optimal_block(
                len(data), reblock_data)[0]
            opt_block = reblock_data[opt]
            be = opt_block[4]
        else:
            be = 0
        ave.append(average)
        block_errors.append(be)

    ave_bf = np.asarray(ave)/bound_frac
    be_bf = np.asarray(block_errors)/bound_frac
    return ave_bf, be_bf


def get_blockerror(Data):
    data = Data
    average = np.average(data)
    be = block(data)**.5
    return average, be


def get_blockerror_pyblock(Data):
    average = np.average(Data)
    if (average != 0) and (average != 1):
        reblock_data = pyblock.blocking.reblock(Data)
        opt = pyblock.blocking.find_optimal_block(len(Data), reblock_data)[0]
        be = reblock_data[opt][4]
    else:
        be = 0
    return average, float(be)


def get_blockerror_pyblock_nanskip(Data):
    average = np.average(Data)
    # print(average,Data,len(Data))
    if (average != 0) and (average != 1):
        reblock_data = pyblock.blocking.reblock(Data)
        opt = pyblock.blocking.find_optimal_block(len(Data), reblock_data)[0]
        # print(opt)
        # print(math.isnan(opt))
        if(math.isnan(opt)):
            be_max = 0
            for i in range(0, len(reblock_data)):
                be = reblock_data[i][4]
                if(be > be_max):
                    be_max = be
        else:
            be = reblock_data[opt][4]
    else:
        be = 0
    return average, float(be)


def calc_Sa(traj, helixBB):
    trjBB = traj
    BB = trjBB.topology.select("name CA")
    HBB = helixBB.topology.select("name CA")

    trjBB.restrict_atoms(BB)
    helixBB.restrict_atoms(HBB)
    trjBB.center_coordinates()
    helixBB.center_coordinates()

    RMS_start = 1
    RMS_stop = 51
    RMS = []
    for i in range(RMS_start, RMS_stop):
        sel = helixBB.topology.select(
            "residue %s to %s and backbone" % (i, i+6))
        rmsd = md.rmsd(trjBB, helixBB, atom_indices=sel)
        RMS.append(rmsd)
    RMS = np.asarray(RMS)

    Sa = (1.0-(RMS/0.10)**8)/(1-(RMS/0.10)**12)
    Sa_total = np.sum(Sa, axis=0)

    return Sa_total


def calc_rg(traj):
    mass = []
    for at in traj.topology.atoms:
        mass.append(at.element.mass)
    mass_CA = len(mass)*[0.0]
    # put the CA entries equal to 1.0
    for i in traj.topology.select("name CA"):
        mass_CA[i] = 1.0
    # calculate CA radius of gyration
    rg_CA = md.compute_rg(traj, masses=np.array(mass_CA))

    return rg_CA


def make_smooth(x, y):
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(xnew)
    return xnew, y_smooth


def get_hist(array):
    histo = np.histogram(array, bins=20)
    norm = histo[0]/np.sum(histo[0])
    x = histo[1][0:-1]
    y = norm
    X1_smooth, Y1_smooth = make_smooth(x, y)
    return X1_smooth, Y1_smooth


def contact_map(traj):
    contact_maps = []
    for i in range(1, 57):
        contact_map = []
        for j in range(1, 57):
            if i == j:
                contacts = 0
            else:
                dist = md.compute_contacts(traj, [[i, j]])
                array = np.asarray(dist[0]).astype(float)
                distance = np.average(array)
                contact = np.where(array < 1.2, 1, 0)
                contacts = np.average(contact)
            contact_map.append(contacts)
        contact_maps.append(contact_map)
    final_map = np.asarray(contact_maps).astype(float)

    return final_map

def calc_contact_map_CA(traj):
    contact_maps = []
    for i in range(0, 56):
        contact_map = []
        for j in range(0, 56):
            if i == j:
                contacts = 0
            else:
                dist = md.compute_distances(traj,[[i,j]])
                array = dist.reshape(1, dist.shape[0])[0]
                contact = np.where(array < 1.2, 1, 0)
                contacts = np.average(contact)
            contact_map.append(contacts)
        contact_maps.append(contact_map)
    final_map = np.asarray(contact_maps).astype(float)
    
    return final_map    
def plot_Sa_rg(rg_CA, Sa_total):
    a, xedges, yedges = np.histogram2d(rg_CA, Sa_total, 30, [
        [0.9, 2.5], [0, 25.0]], normed=True, weights=None)
    a = np.log(np.flipud(a)+.000001)
    T = 300
    a = -(0.001987*T)*a

    im = plt.imshow(a, interpolation='gaussian', extent=[
                    yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap='jet', aspect='auto')
    cbar_ticks = [0, 1, 2, 3, 4, 5]
    cb = plt.colorbar(ticks=cbar_ticks, format=('% .1f'),
                      aspect=10)  # grab the Colorbar instance
    imaxes = plt.gca()
    plt.xlim(0, 24.9)
    plt.ylabel("Radius of Gryation", size=35, labelpad=15)
    plt.xlabel(r'S$\alpha$', size=35, labelpad=15)
    plt.xticks(size='26')
    plt.yticks(size='26')
    plt.axes(cb.ax)
    plt.clim(vmin=0.1, vmax=3.0)
    plt.tight_layout()


def get_Sa_rg_hist(rg_CA, Sa_total):
    a, xedges, yedges = np.histogram2d(rg_CA, Sa_total, 30, [
        [0.9, 2.5], [0, 25.0]], normed=True, weights=None)
    a = np.log(np.flipud(a)+.000001)
    T = 300
    a = -(0.001987*T)*a

    return a, xedges, yedges

def calculate_tsne(name, traj):
    outdir = workdir + name + '/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    rmsd = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        rmsd[i] = md.rmsd(traj, traj, i)
    print('Max pairwise rmsd: %f nm' % np.max(rmsd))
    rmsd_sym = suv.check_symmetric(rmsd, raise_exception=False)

    with open(outdir + 'status.txt', 'a') as f1:
        f1.write("\n")
        print('symmetry check completed', file=f1)

    # Kmeans clustering
    range_n_clusters = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    perplexityVals = range(100, 2100, 100)

    # Creating the TSNE object and projection
    perplexityVals = range(100, 2100, 100)

    for i in perplexityVals:
        tsneObject = TSNE(n_components=2, perplexity=i, early_exaggeration=10.0, learning_rate=100.0, n_iter=3500,
                          n_iter_without_progress=300, min_grad_norm=1e-7, metric="precomputed", init='random', method='barnes_hut', angle=0.5)
        # metric is precomputed RMSD distance. if you provide Raw coordinates, the TSNE will compute the distance by default with Euclidean metrics
        tsne = tsneObject.fit_transform(rmsd_sym)
        np.savetxt(outdir + "tsnep{0}".format(i), tsne)

    for perp in perplexityVals:
        tsne = np.loadtxt(outdir + 'tsnep'+str(perp))
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters).fit(tsne)
            np.savetxt(outdir + 'kmeans_'+str(n_clusters)+'clusters_centers_tsnep' +
                       str(perp), kmeans.cluster_centers_, fmt='%1.3f')
            np.savetxt(outdir + 'kmeans_'+str(n_clusters)+'clusters_tsnep' +
                       str(perp)+'.dat', kmeans.labels_, fmt='%1.1d')
    # Compute silhouette score based on low-dim and high-dim distances
            silhouette_ld = silhouette_score(tsne, kmeans.labels_)
            np.fill_diagonal(rmsd_sym, 0)
            silhouette_hd = metrics.silhouette_score(
                rmsd_sym, kmeans.labels_, metric='precomputed')
            with open(outdir + 'silhouette.txt', 'a') as f:
                f.write("\n")
                print(perp, n_clusters, silhouette_ld, silhouette_hd,
                      silhouette_ld*silhouette_hd, file=f)


