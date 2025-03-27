#! usr/bin/env python
import os
import copy
import math
import time
import torch
import shutil
import numpy as np
from pynep.io import load_nep
import torch.distributed as dist
from torch.multiprocessing import Process
from ase.neighborlist import neighbor_list
from ase.io import read, write, Trajectory 

def init_GPBBProcess_v1(rank, size, fn, 
                        images,
                        outname,
                        elements_dict,
                        scale_factors,
                        r_list,
                        sc_mult,
                        tolerance,
                        confidence,
                        steps,
                        step_size,
                        start,
                        backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, images, outname, elements_dict, scale_factors, r_list, sc_mult, tolerance, confidence, steps, step_size, start)
    
def init_GPBBProcess_v2(rank, size, fn,
                        images, 
                        outname,
                        elements_dict,
                        scale_factors,
                        cutoff,
                        tolerance,
                        correlation,
                        steps,
                        step_size,
                        start,
                        backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, images, outname, elements_dict, scale_factors, cutoff, tolerance, correlation, steps, step_size, start)
  
def run_GPBB_v1(fname, num_core, element_dict, scale_factors, r_list, sc_mult, tolerance, confidence, steps, step_size):
    rawdir = os.path.dirname(fname) + '/adjusted_traj'
    if os.path.exists(rawdir):
        shutil.rmtree(rawdir)
    os.mkdir(rawdir)
    print('=======================================================')
    print(
        'Starting a GPBB Algorithm job at %s',
        time.strftime('%X %x %Z'))
    images = read(fname, ':')
    nimages = len(images)
    print('  Total number of traning images: %d', nimages)
    print('=======================================================')
    total_of_cores = num_core
    processes= []
    num_each_process = math.ceil(nimages / total_of_cores)

    trajs = []
    for rank in range(total_of_cores):
        start = num_each_process * rank
        end = start + num_each_process
        trajf = os.path.join(rawdir, f'{rank}_{start}_{end}.traj')
        p = Process(target=init_GPBBProcess_v1,
                    args=(rank, total_of_cores, GPBB_v1, images[start:end], 
                          trajf, element_dict, scale_factors, r_list, sc_mult, 
                          tolerance, confidence, steps, step_size, start))
        trajs.append(trajf)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    combined_images = []
    for f in trajs:
        structures = read(f, ':')
        combined_images.extend(structures)
    write(f'{rawdir}/Adjusted.xyz', combined_images, format='extxyz')

def run_GPBB_v2(fname, num_core, element_dict, scale_factors, cutoff, tolerance, correlation, steps, step_size):
    rawdir = os.path.dirname(fname) + '/adjusted_traj'
    if os.path.exists(rawdir):
        shutil.rmtree(rawdir)
    os.mkdir(rawdir)
    print('=======================================================')
    print(
        'Starting a GPBB Algorithm job at %s',
        time.strftime('%X %x %Z'))
    images = read(fname, ':')
    nimages = len(images)
    print('  Total number of traning images: %d', nimages)
    print('=======================================================')
    total_of_cores = num_core
    processes= []
    num_each_process = math.ceil(nimages / total_of_cores)
    

    trajs = []
    for rank in range(total_of_cores):
        start = num_each_process * rank
        end = start + num_each_process
        trajf = os.path.join(rawdir, f'{rank}_{start}_{end}.traj')
        p = Process(target=init_GPBBProcess_v2,
                    args=(rank, total_of_cores, GPBB_v2, images[start:end], 
                          trajf, element_dict, scale_factors, cutoff, 
                          tolerance, correlation, steps, step_size, start))
        trajs.append(trajf)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    combined_images = []
    for f in trajs:
        structures = read(f, ':')
        combined_images.extend(structures)
    write(f'{rawdir}/Adjusted.xyz', combined_images, format='extxyz')
    
def GPBB_v1(rank, size, images, outname, elements_dict, scale_factors, r_list, sc_mult, tolerance, confidence, steps, step_size, start, vac=8):
    for i, image in enumerate(images):
        symbols = image.symbols.species()
        num = len(symbols)
        if num == 1:
            image_scale = Single_element_bulk_adjust(image, elements_dict, scale_factors)
            with open(f'process-{rank}.log', 'a') as f:
                print('Single element bulk image %s: Done'% (i+start), file=f)
            with Trajectory(outname, 'a') as traj:
                traj.write(image_scale)
        elif num > 1:
            pbc_ = _vacuum_judge(image)
            
            if np.sum(pbc_) == 3:
                img_nopbc = BL_adjust(rank, image, elements_dict, steps, scale_factors, tolerance, step_size, confidence)
                img_nopbc.center(vacuum=vac, axis=(0, 1, 2))
                with open(f'process-{rank}.log', 'a') as f:
                    print('Non pbc image %s: Done'% (i+start), file=f)
                with Trajectory(outname, 'a') as traj:
                    traj.write(img_nopbc)
            else:
                img_mut, r = adjust_mult_bulk_surface(image, elements_dict, r_list, sc_mult)
                with open(f'process-{rank}.log', 'a') as f:
                    print('Multi-elements image %s ratio is %s: Done'% (i+start, r), file=f)
                with Trajectory(outname, 'a') as traj:
                    traj.write(img_mut)
                    
def GPBB_v2(rank, size, images, outname, elements_dict, scale_factors, cutoff, tolerance, correlation, steps, step_size, start, vac=8):
    for i, image in enumerate(images):
        symbols = image.symbols.species()
        num = len(symbols)
        if num == 1:
            image_scale = Single_element_bulk_adjust(image, elements_dict, scale_factors)
            print('Single element bulk image %s: Done'% (i+start))
            with Trajectory(outname, 'a') as traj:
                traj.write(image_scale)
        elif num > 1:
            pbc_ = _vacuum_judge(image)
            if np.sum(pbc_) == 0:
                img_nopbc = Local_env_adjust(image, elements_dict, cutoff, steps, scale_factors, tolerance, correlation, step_size)
                img_nopbc.center(vacuum=vac, axis=(0,1,2))
                print('Non pbc image %s: Done'% (i+start))
                with Trajectory(outname, 'a') as traj:
                    traj.write(img_nopbc)
            else:
                img_mut= Local_env_adjust(image, elements_dict, cutoff, steps, scale_factors, tolerance, correlation, step_size)
                print('The Multi-elements image %s: Done'% (i+start))
                with Trajectory(outname, 'a') as traj:
                    traj.write(img_mut)               
                
# TODO: Local environment have changed. We need think about a general method to avoid losting neighborlist.
# SOLUTION: target_neighbor_list changed in each step.
def Local_env_adjust(image, exchange_e, cutoff, steps, sc, tolerance, correlation, step_size, bin_range=[1.5, 6]): 
    # Get all types of symbols and the corresponding index for each image.
    symbols = image.symbols.species()
    image_t = copy.deepcopy(image)
    pos = image.get_positions()
    dict_symbols = image_t.symbols.indices()
    
    # Exchange symbols 
    for i in symbols:
        image_t.symbols[dict_symbols[i]] = exchange_e[i]
    '''
    Initialize of neighborlist. 
    '''
    neighbors_bak = all_neighbors, id1, id2, shift_vectors = neighbor_list('dijS', image_t, cutoff)
    target_neighbors = np.zeros(np.shape(all_neighbors))
    # Get the corresponding atom id of each variable from all_neighbors.    
    print('STEP\tMOVED_NUM\tR2\tCONFIDENCE')
    # A variable in the for loop. 
    # These variables will be updated in each step. 
    pos_loop = pos
    # 1. Define the target_all_neighbors.    
    for j in range(len(neighbors_bak[0])):
        scale_type = [image_t[neighbors_bak[1][j]].symbol, image_t[neighbors_bak[2][j]].symbol]
        if image_t[neighbors_bak[1][j]].symbol != image_t[neighbors_bak[2][j]].symbol:
            scale_type = sorted(scale_type) 
        scale_factor = sc[f"{scale_type[0]}-{scale_type[1]}"]
        target_neighbors[j] = neighbors_bak[0][j] * scale_factor
        
    ini_bins = np.arange(bin_range[0], bin_range[1], (bin_range[1]-bin_range[0])/100)
    h, bins = np.histogram(target_neighbors, bins=ini_bins) 
    pdf = h/(4*np.pi/3*(bins[1:]**3 - bins[:-1]**3)) * image.get_volume()/len(image)

    # Start for loop
    for step in range(1, steps+1):    
        # 1. Get the R2 between adjusted image pdf and target pdf.
        h_p, bins_p = np.histogram(neighbors_bak[0], bins=ini_bins)
        pdf_p = h_p/(4*np.pi/3*(bins_p[1:]**3 - bins_p[:-1]**3)) * image_t.get_volume()/len(image_t)
        ss_res = np.sum((pdf - pdf_p) ** 2)   # Residual sum of squares
        ss_tot = np.sum((pdf - np.mean(pdf)) ** 2)  # Total sum of squares
        all_dif = np.abs(target_neighbors - neighbors_bak[0])
        max_dif = np.max(all_dif)
        #TODO: Add confidence or tolerance ?
        confidence = len(all_dif[all_dif <= tolerance]) / len(all_dif.reshape(-1)) 
        R2 = 1 - (ss_res / ss_tot)
        
        # 2. Local environment adjustment.
        if R2 <= correlation:
            # Initialize the moved bond length matrix 
            movea = np.zeros(np.shape(pos_loop))
            # Check each bond length
            moved = 0
            for n in range(len(neighbors_bak[0])):
                # Attention: The direction of vector is from id1[n] to id2[n]
                d = pos_loop[id2[n]] - pos_loop[id1[n]] + np.dot(shift_vectors[n], image_t.get_cell())
                mould_l = np.sqrt(np.vdot(d, d))
                uni_vec = d / mould_l
                esl = step_size * uni_vec
                dd = target_neighbors[n] - neighbors_bak[0][n]
                # print(dd)
                if np.abs(dd) >= tolerance:
                    if dd > 0:
                        movea[id1[n]] -= esl
                        movea[id2[n]] += esl
                    elif dd < 0:
                        movea[id1[n]] += esl/2
                        movea[id2[n]] -= esl/2
                    moved += 1
                else:
                        movea[id1[n]] += 0
                        movea[id2[n]] -= 0
            pos_loop += movea
            image_t.set_positions(pos_loop)
            for n in range(len(neighbors_bak[0])):
                d = pos_loop[id2[n]] - pos_loop[id1[n]] + np.dot(shift_vectors[n], image_t.get_cell())
                mould_l1 = np.sqrt(np.vdot(d, d))
                # print(neighbors_bak[0][n])
                neighbors_bak[0][n] = mould_l1
                # print(neighbors_bak[0][n], target_neighbors[n])
            all_dif = np.abs(target_neighbors - neighbors_bak[0])
            max_dif = np.max(all_dif)
            #TODO: Add confidence or tolerance ?
            confidence = len(all_dif[all_dif <= tolerance]) / len(all_dif.reshape(-1)) 
            # Update all neighborlist parameters.
            all_neighbors_temp = neighbor_list('d', image_t, cutoff)
            # When it comes to the last step
            h_p, bins_p = np.histogram(all_neighbors_temp, bins=ini_bins)
            pdf_p = h_p/(4*np.pi/3*(bins_p[1:]**3 - bins_p[:-1]**3)) * image_t.get_volume()/len(image_t)
            ss_res = np.sum((pdf - pdf_p) ** 2)   # Residual sum of squares
            ss_tot = np.sum((pdf - np.mean(pdf)) ** 2)  # Total sum of squares
            R2 = 1 - (ss_res / ss_tot)
            print(f'{step}\t{moved}\t{R2}\t{confidence}')
        else:
            print('[INFO]: Reached required accuracy')
            break
    return image_t


def BL_adjust(rank, image, exchange_e, steps, sc, tolerance, step_size, confidence): # Now it only suitable for pbc=False structures.
    # Get all types of symbols and the corresponding index for each image.
    symbols = image.symbols.species()
    image_t = copy.deepcopy(image)
    pos = image.get_positions()
    dict_symbols = image_t.symbols.indices()
    # Exchange symbols 
    for i in symbols:
        image_t.symbols[dict_symbols[i]] = exchange_e[i]

    # Get all of bond lengths for the image. Now this method is only suitable for non-pbc structures. 
    # TODO: Adding pbc for multi-element bulk.
    pos_tensor = torch.tensor(pos)
    BL_matrix = torch.cdist(pos_tensor, pos_tensor)
    BL_matrix_target = BL_matrix
    # obtain target bond length matrix
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            scale_type = [image_t[i].symbol, image_t[j].symbol]
            if image_t[i].symbol != image_t[j].symbol:
                scale_type = sorted(scale_type)    
            scale_factor = sc[f"{scale_type[0]}-{scale_type[1]}"]
            BL_matrix_target[i][j] = scale_factor * BL_matrix[i][j]
    # Bond length adjustment.
    # 1. Initialize the total step when running BL adjustment.
    with open(f'process-{rank}.log', 'a') as f:
        print('STEP\tMOVED_NUM\tMAX_DIFF\tCONFIDENCE_LEVEL\tRMSE', file=f)
    pos_loop = pos
    for step in range(1, steps+1): 
        pos_loop_tensor = torch.tensor(pos_loop)
        # Change the difference as absolute value
        all_dif = np.abs(BL_matrix_target - torch.cdist(pos_loop_tensor, pos_loop_tensor))
        max_dif = torch.max(all_dif)
        # print(max_dif)
        score = len(all_dif[all_dif < tolerance]) / len(all_dif.reshape(-1))
        rmse = np.sqrt(np.sum(np.square(score))/ len(all_dif.reshape(-1)))
        if score <= confidence :
            # Initialize the moved bond length matrix 
            movea = np.zeros(np.shape(pos_loop))
            # Check each bond length
            moved = 0
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    d = pos_loop[i] - pos_loop[j]
                    mould_l = np.sqrt(np.vdot(d, d))
                    uni_vec = d / mould_l
                    dd = BL_matrix_target[i][j] - mould_l
                    esl = step_size * uni_vec
                    if np.abs(dd) > tolerance:
                        if dd > 0:
                            movea[i] += esl
                            movea[j] -= esl
                        elif dd < 0:
                            movea[i] -= esl/2
                            movea[j] += esl/2
                        moved += 1
                    else:
                        continue
            pos_loop += movea
            pos_loop_tensor = torch.tensor(pos_loop)
            all_dif = np.abs(BL_matrix_target - torch.cdist(pos_loop_tensor, pos_loop_tensor))
            max_dif = torch.max(all_dif)
            score = len(all_dif[all_dif <= tolerance]) / len(all_dif.reshape(-1))
            with open(f'process-{rank}.log', 'a') as f:
                print(f'{step}\t{moved}\t{max_dif}\t{score}\t{rmse}', file=f)
        else:
            with open(f'process-{rank}.log', 'a') as f:
                print('[INFO]: Reached required accuracy', file=f)
            break
    image_t.set_positions(pos_loop)
    return image_t

def Single_element_bulk_adjust(image, exchange_e, sc):
    cell0 = image.get_cell()
    image_t = copy.deepcopy(image)
    symbol = image.symbols.species()
    if len(symbol) == 1: 
        e = list(symbol)[0]
        ec = exchange_e[e]
        scale_factor = sc[f"{ec}-{ec}"]
        image_t.symbols[:] = ec
        cell1 = cell0 * scale_factor
        image_t.set_cell(cell=cell1, scale_atoms=True)
    else:
        raise TypeError(f'Error! Image {image} is not a single element bulk.')
    return image_t
def ratio_find(images):
    ratio = []
    for id, img in enumerate(images):
        symbols = img.symbols.species()
        num = len(symbols)
        if num > 1:
            pbc_ = _vacuum_judge(img)
            if np.sum(pbc_) == 3:
                ele_dict = img.symbols.indices()
                ele_num = [len(i) for i in list(ele_dict.values())]
                r = np.round([(ele_num[i] / len(img)) for i in range(len(ele_dict))], 3).reshape(-1)
                ratio.append(r)
    a = np.unique(ratio, axis=0)
    return a


def adjust_mult_bulk_surface(img, exchange_e, ratio_list, scale_factors):
    inicell = img.get_cell()
    newi = copy.deepcopy(img)
    symbols_ = img.symbols.species()
    num_ = len(symbols_)
    dic_ele = img.symbols.indices()
    ele_num = [len(i) for i in list(dic_ele.values())]
    r = np.round([(ele_num[i] / len(img)) for i in range(len(dic_ele))], 3).reshape(-1)
    for i in symbols_:
        newi.symbols[dic_ele[i]] = exchange_e[i]
    if np.any(np.isin(ratio_list, r)):
        j = np.where(ratio_list == r)
        if len(j) == num_:
            j = j[0][0]
        newcell = []
        for id, ll in enumerate(inicell):
            newcell.append(ll * scale_factors[j][id])
        newi.set_cell(cell=newcell, scale_atoms=True)
    else:
        raise ValueError(f'Can\'t find the ratio {r} corresponding scale factor.' )
    return newi, r

def Find_Bulk_surface_ratio(inputfname):
    imgs = read(inputfname, ':')
    ratio_list = ratio_find(imgs)
    print("Ratio of multi-element bulk in total structure is ", ratio_list)
    print(f'You need set {len(ratio_list)} parameters at SC_MULT.')
    return ratio_list


def pbc(r, box, ibox = None, direct=True):
    """
    Applies periodic boundary conditions.
    Parameters:
        r:      the vector the boundary conditions are applied to
        box:    the box that defines the boundary conditions
        ibox:   the inverse of the box. This will be calcluated if not provided.
    """
    if ibox is None:
        ibox = np.linalg.inv(box).T
    vdir = np.dot(r, ibox)
    vdir = (vdir % 1.0 + 1.5) % 1.0 - 0.5
    if direct == True:
        return vdir
    else:
        return np.dot(vdir, box)

def _vacuum(image): 
    cellpar = image.cell.cellpar()
    coord = image.get_positions()
    box = image.get_cell()
    ibox = np.linalg.inv(box.T)
    pbc_coord = pbc(coord, box, ibox, direct=True)
    dir_coord = [pbc_coord[:, i] for i in range(3)]
    v = [np.max(i) - np.min(i) for i in dir_coord]
    pos2 = np.dot(coord, ibox)
    return [cellpar[i]-v[i]*cellpar[i] for i in range(3)]

def _vacuum_judge(image, vac=3):
    return np.array(_vacuum(image)) < vac

if __name__ == '__main__':

 ########################################- Parameters of code -################################################  
    VERSION = 1.0
    dir = '/work/mse-wangjq/work/ago/1818'
    FILENAME = dir + '/PdO1818.xyz'
    NUM_OF_CORE = 48
    ELEMENTS = {'Pd':'Ag', 'O':'O'}
    # Sort by alphabetical order when the bond with different elements
    SCALE_FACTOR = {'Ag-Ag': 1.049,
                    'Ag-O': 1.052,
                    'O-O': 1.000}
    SC_MULT = [[1.049, 1.052, 1.042]]  # Only available for version 1.
    CUTOFF = 6         # Only available for version 2. Suggestion: Set the same values when training.
    TOLERENCE = 0.01    # Average bond length error.
    CONFIDENCE_LEVEL = 0.95   # How many bond length reached the torlence.
    STEP = 3000
    BIN_SIZE = 0.0015  # Bond length change in each step.
 #############################################################################################################   
    
    
    '''
Version 1 code 
    '''
    if VERSION == 1.0:
##########################-Step 1-############################################################################
        MUTI_ELE_LIST = Find_Bulk_surface_ratio(FILENAME)
        if len(SC_MULT) == len(MUTI_ELE_LIST):
##########################-Step 2-############################################################################
            run_GPBB_v1(FILENAME, NUM_OF_CORE, ELEMENTS, SCALE_FACTOR, MUTI_ELE_LIST, SC_MULT, TOLERENCE, CONFIDENCE_LEVEL, STEP, BIN_SIZE)
        else:
            raise ValueError(f'The right number of SC_MULT is {len(MUTI_ELE_LIST)}, but you only set {len(SC_MULT)}!\nPlease set parameters follow the queue of multi-element bulk ratio {MUTI_ELE_LIST}.')
    elif VERSION == 2.0:
        run_GPBB_v2(FILENAME, NUM_OF_CORE, ELEMENTS, SCALE_FACTOR, CUTOFF, TOLERENCE, CONFIDENCE_LEVEL, STEP, BIN_SIZE)


