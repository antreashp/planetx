
import numpy as np
# import SAT
from itertools import combinations,product
from pysat.solvers import Glucose3,Lingeling, Minisat22,MapleCM,Glucose42
from pysat.formula import CNF

def generate_single_sector_clauses(sector):
    clauses = []
    objects = list(range(1, 8))  # 6 objects numbered from 1 to 6
    
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            # If object i is in sector, object j cannot be in the same sector
            clauses.append((-1000 - (sector * 10) - objects[i], -1000 - (sector * 10) - objects[j]))
            # Vice versa
            clauses.append((-1000 - (sector * 10) - objects[j], -1000 - (sector * 10) - objects[i]))

    return clauses

def generate_clauses_for_all_sectors():
    all_clauses = []
    for sector in range(1, 19):  # 18 sectors
        all_clauses.extend(generate_single_sector_clauses(sector))
    
    return all_clauses

ALLOWED_SECTORS_FOR_COMETS = [2,3,5,7,11,13,17]

def generate_optimized_cloud_ofkera_adjacency_clauses():
    # Assuming Clouds have the object ID of 3 and Ofkera has the object ID of 2
    cloud_id = 3
    ofkera_id = 2
    clauses = []

    for i in range(1, 19):
        prev_i = i - 1 if i > 1 else 18
        next_i = i + 1 if i < 18 else 1
        
        # Constructing the literals with the correct format
        cloud_literal = -1000 - (i * 10) - cloud_id
        prev_ofkera_literal = 1000 + (prev_i * 10) + ofkera_id
        next_ofkera_literal = 1000 + (next_i * 10) + ofkera_id
        
        clause = [cloud_literal, prev_ofkera_literal, next_ofkera_literal]
        clauses.append(clause)

    return clauses

   
    
def generate_asteroid_adjacency_clauses():
    # Assuming Asteroids have the object ID of 5
    asteroid_id = 5
    clauses = []

    for i in range(1, 19):
        prev_i = i - 1 if i > 1 else 18
        next_i = i + 1 if i < 18 else 1

        # Constructing the literals with the correct format
        asteroid_literal = -1000 - (i * 10) - asteroid_id
        prev_asteroid_literal = 1000 + (prev_i * 10) + asteroid_id
        next_asteroid_literal = 1000 + (next_i * 10) + asteroid_id
        
        clause = [asteroid_literal, prev_asteroid_literal, next_asteroid_literal]
        clauses.append(clause)

    return clauses
def generate_comet_location_clauses():
    # Assuming Comets have the object ID of 6
    comet_id = 6
    allowed_sectors = {2, 3, 5, 7, 11, 13, 17}
    clauses = []

    for i in range(1, 19):
        if i not in allowed_sectors:
            # Constructing the negative literals for disallowed sectors
            disallowed_comet_literal = -1000 - (i * 10) - comet_id
            clauses.append([disallowed_comet_literal])

    return clauses
def generate_planetx_dwarf_nonadjacency_clauses():
    # Assuming PlanetX has the object ID of 1 and Dwarfs have the object ID of 4
    planetx_id = 1
    dwarf_id = 4
    clauses = []

    for i in range(1, 19):
        prev_i = i - 1 if i > 1 else 18
        next_i = i + 1 if i < 18 else 1

        # Constructing the literals with the correct format
        planetx_literal = 1000 + (i * 10) + planetx_id
        prev_dwarf_literal = -1000 - (prev_i * 10) - dwarf_id
        next_dwarf_literal = -1000 - (next_i * 10) - dwarf_id

        # If PlanetX is in sector i, then the previous and next sectors cannot have a dwarf
        clauses.append([-planetx_literal, prev_dwarf_literal])
        clauses.append([-planetx_literal, next_dwarf_literal])

    return clauses
def generate_each_sector_has_object():
    clauses = []
    for sector in range(1, 19):
        clause = []
        for object_id in range(1, 8):  # Looping over the 6 objects.
            clause.append(sector * 10 + 1000 + object_id)
        clauses.append(clause)
    return clauses


def generate_cnf_for_n_objects(n,obj_id):
    cnf_clauses = []
    sectors = [f"{i:02}" for i in range(1, 19)]
    object_repr = str(obj_id)

    # 1. For any combination of n+1 sectors, at least one of them should be empty.
    for combination in combinations(sectors, n+1):
        clause = [f"-1{s}{object_repr}" for s in combination]
        cnf_clauses.append(clause)

    # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
    for combination in combinations(sectors, 18-n+1):
        clause = [f"1{s}{object_repr}" for s in combination]
        cnf_clauses.append(clause)
    cnf_clauses = [[int(float(j)) for j in i] for i in cnf_clauses] 
    return cnf_clauses


def dwarf_custom():
    
    sectors = [f"{i:02}" for i in range(1, 19)]

    clauses = []
    



    
    for sector in range(1, 19):
        forward_window = int(sectors[(sector+4)%18])
        backward_window = int(sectors[(sector-6)%18])
        clauses.append([-(1000+ sector * 10  + 4),(1000+ forward_window * 10  + 4),(1000+ backward_window * 10  + 4)])
        clauses.append([-(1000+ forward_window * 10  + 4),-(1000+ backward_window * 10  + 4)])
    
    return clauses

def generate_exact_distance_constraints():
    clauses = []

    
    for i in range(1, 19):
        next_i = (i + 5) % 18
        if next_i == 0:  # Handling the wrap-around case
            next_i = 18

        # If object 4 is in sector i, then it's also in sector (i+5)%18
        clauses.append([f"-1{i:02}4", f"1{next_i:02}4"])

        # If object 4 is not in sector i, then it's also not in sector (i+5)%18
        clauses.append([f"1{i:02}4", f"-1{next_i:02}4"])
    clauses = [[int(float(j)) for j in i] for i in clauses]
    return clauses
def generate_cnf_for_object_7():
    sectors = [f"{i:02}" for i in range(1, 19)]

    clauses = []
    
    for sector in range(1, 19):
        between_window = [int(sectors[(sector+j)%18]) for j in range(0, 4)]
        combs = combinations(between_window, 3)
        # print(between_window)
        for comb in combs:
            clauses.append([-(1000+ sector * 10  + 4),-(1000+ ((sector+5)%18) * 10  + 4),(1000+ (comb[0]) * 10  + 7),(1000+ (comb[1]) * 10  + 7),(1000+ (comb[2]) * 10  + 7)])

    return clauses


def target(sector,body):
    if body == 2 :
        return [int('1'+str(sector).zfill(2)+str(2)),int('1'+str(sector).zfill(2)+str(1))]
    
    
    if body != 4 and body !=7:
        return [int('1'+str(sector).zfill(2)+str(body))]
    else:
        return [int('1'+str(sector).zfill(2)+str(4)),int('1'+str(sector).zfill(2)+str(7))]

def survey(starting,ending, body,n):
    cnf_clauses = []
    sectors = [f"{i:02}" for i in range(starting, ending+1)]
    if body ==2:
        if starting > ending:
            dist =  19 -starting+ ending
        else:
            dist = -starting+ ending
        cnf_clauses = generate_cnf_for_objects_1_and_2(starting, dist-1, n)
        cnf_clauses = [[int(float(j)) for j in i] for i in cnf_clauses] 


        return cnf_clauses
    if body == 4 and n==0:
        for sector in sectors:
            cnf_clauses.append([int('-1'+str(sector)+str(4))])
            cnf_clauses.append([int('-1'+str(sector)+str(7))])
        return cnf_clauses
    elif body == 4 and n==1:
        for sector in sectors:
            cnf_clauses.append([int('-1'+str(sector).zfill(2)+str(4)),int('-1'+str(sector).zfill(2)+str(7))])
        return cnf_clauses
    elif body == 4 and n==2:
        cnf_clauses = generate_cnf_for_object_7_and_4_in_range(starting, ending+1)
            
        cnf_clauses = [[int(float(j)) for j in i] for i in cnf_clauses] 
        return cnf_clauses
    elif body == 4 and n==3:
        object_repr = str(body)

        # 1. For any combination of n+1 sectors, at least one of them should be empty.
        for combination in combinations(sectors, 2+1):
            clause = [f"-1{s}{str(7)}" for s in combination]
            cnf_clauses.append(clause)

        # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
        for combination in combinations(sectors, ending+1-starting-2+1):
            
            clause = [f"1{s}{str(7)}" for s in combination]
            
            cnf_clauses.append(clause)

        # 1. For any combination of n+1 sectors, at least one of them should be empty.
        for combination in combinations(sectors, 1+1):
            clause = [f"-1{s}{str(4)}" for s in combination]
            cnf_clauses.append(clause)

        # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
        for combination in combinations(sectors, ending+1-starting-1+1):
            
            clause = [f"1{s}{str(4)}" for s in combination]
            
            cnf_clauses.append(clause)


        cnf_clauses = [[int(float(j)) for j in i] for i in cnf_clauses] 


        return cnf_clauses
    elif body == 4 and n==3:
    
        object_repr = str(body)

        # 1. For any combination of n+1 sectors, at least one of them should be empty.
        for combination in combinations(sectors, 2+1):
            clause = [f"-1{s}{str(7)}" for s in combination]
            cnf_clauses.append(clause)

        # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
        for combination in combinations(sectors, ending+1-starting-2+1):
            
            clause = [f"1{s}{str(7)}" for s in combination]
            
            cnf_clauses.append(clause)

        # 1. For any combination of n+1 sectors, at least one of them should be empty.
        for combination in combinations(sectors, 1+1):
            clause = [f"-1{s}{str(4)}" for s in combination]
            cnf_clauses.append(clause)

        # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
        for combination in combinations(sectors, ending+1-starting-1+1):
            
            clause = [f"1{s}{str(4)}" for s in combination]
            
            cnf_clauses.append(clause)


        cnf_clauses = [[int(float(j)) for j in i] for i in cnf_clauses] 


        return cnf_clauses
    elif body == 4 and n==4:
    
        object_repr = str(body)

        # 1. For any combination of n+1 sectors, at least one of them should be empty.
        for combination in combinations(sectors, 2+1):
            clause = [f"-1{s}{str(7)}" for s in combination]
            cnf_clauses.append(clause)

        # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
        for combination in combinations(sectors, ending+1-starting-2+1):
            
            clause = [f"1{s}{str(7)}" for s in combination]
            
            cnf_clauses.append(clause)

        # 1. For any combination of n+1 sectors, at least one of them should be empty.
        for combination in combinations(sectors, 2+1):
            clause = [f"-1{s}{str(4)}" for s in combination]
            cnf_clauses.append(clause)

        # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
        for combination in combinations(sectors, ending+1-starting-2+1):
            
            clause = [f"1{s}{str(4)}" for s in combination]
            
            cnf_clauses.append(clause)


        cnf_clauses = [[int(float(j)) for j in i] for i in cnf_clauses] 


        return cnf_clauses
    # sectors = [f"{i:02}" for i in range(starting, ending+1)]
    
    object_repr = str(body)

    # 1. For any combination of n+1 sectors, at least one of them should be empty.
    for combination in combinations(sectors, n+1):
        clause = [f"-1{s}{object_repr}" for s in combination]
        cnf_clauses.append(clause)

    # 2. For any combination of 18-n+1 sectors, at least one of them should contain the object.
    for combination in combinations(sectors, ending+1-starting-n+1):
        
        clause = [f"1{s}{object_repr}" for s in combination]
        
        cnf_clauses.append(clause)
    cnf_clauses = [[int(float(j)) for j in i] for i in cnf_clauses] 
    return cnf_clauses


def generate_cnf_for_object_7_and_4_in_range(start_sector, end_sector):
    clauses = []

    sector_range = list(range(start_sector, end_sector + 1))

    for sector in sector_range:
        # At least one sector within the range should contain object 7
        clause_7 = [f"1{sector:02}7" for sector in sector_range]
        clauses.append(clause_7)
        
        # At least one sector within the range should contain object 4
        clause_4 = [f"1{sector:02}4" for sector in sector_range]
        clauses.append(clause_4)

    for sector in sector_range:
        # For each sector, if object 4 is present, then object 7 shouldn't be present in the same sector
        clauses.append([f"-1{sector:02}4", f"-1{sector:02}7"])

    return clauses
def generate_cyclic_sectors(sector_start, distance, total_sectors=18):
    """Generate sectors in a cyclic manner."""
    sectors = [(sector_start + i) % total_sectors for i in range(distance)]
    sectors = [s if s != 0 else total_sectors for s in sectors]  # converting 0 to total_sectors
    return sectors

def generate_cnf_for_objects_1_and_2(sector_start, distance, num_objects):
    clauses = []
    sectors = generate_cyclic_sectors(sector_start, distance)

    if len(sectors) < num_objects:
        raise ValueError("The range provided doesn't fit the number of objects.")

    # If a sector contains object 2, then it shouldn't contain object 1
    for i in sectors:
        clauses.append([f"-1{i:02}2", f"-1{i:02}1"])

    # Only one instance of object 1 in the range.
    for combo in combinations(sectors, num_objects - 1):
        clause = [f"1{k:02}2" for k in combo]
        remaining_sectors = list(set(sectors) - set(combo))
        for r in remaining_sectors:
            clause.append(f"-1{r:02}2")
        clause.append(f"1{remaining_sectors[0]:02}1")  # using the first sector not in combo for object 1
        clauses.append(clause)

    # Not more than one sector should contain object 1
    for i in sectors:
        for j in sectors:
            if i != j:
                clauses.append([f"-1{i:02}1", f"-1{j:02}1"])

    return clauses
custom_dwarfs = dwarf_custom()
exact_dwarfs = generate_exact_distance_constraints()
# exit()
indwarfs_rule = generate_cnf_for_object_7()
dwarf_x = generate_planetx_dwarf_nonadjacency_clauses()
comet_locations = generate_comet_location_clauses()
asteroid_adjacent = generate_asteroid_adjacency_clauses()
# dwarf_adjacent = generate_dwarfs_adjacency_clauses()
cloud_ofkera = generate_optimized_cloud_ofkera_adjacency_clauses()

dwarf_number = generate_cnf_for_n_objects(2,4)

x_number = generate_cnf_for_n_objects(1,1)
one_for_sector = generate_clauses_for_all_sectors()
objects_for_sector = generate_each_sector_has_object()
ofkera_number = generate_cnf_for_n_objects(5,2)
comet_number = generate_cnf_for_n_objects(2,6)
indwarfs = generate_cnf_for_n_objects(2,7)
asteroid_number = generate_cnf_for_n_objects(4,5)
clouds_number = generate_cnf_for_n_objects(2,3)




clauses =  one_for_sector +x_number + ofkera_number + clouds_number + asteroid_number + comet_number + objects_for_sector  \
             +comet_locations  +   asteroid_adjacent + cloud_ofkera +dwarf_x +custom_dwarfs + indwarfs + indwarfs_rule


starting_info = [[-1025],[-1033],[-1073],[-1054],[-1057]]
clauses = clauses + starting_info



# target = input()
# targets = [target(1,2),target(2,6),target(3,2),target(4,2),target(5,6),target(6,4),target(7,4),target(8,5),target(9,5),
#            target(10,4),target(11,4),target(12,5),target(13,5),target(14,3),target(15,2),target(16,2),target(17,3),target(18,2),]
# targets = []
# clauses = clauses + targets

# clauses = clauses + survey(1,8, 5,1)
# clauses = clauses + survey(2,8, 5,1)
# clauses = clauses + survey(4,11, 4,2)

# clauses = clauses + survey(4,7, 4,2)
# clauses = clauses + survey(1,6, 4,1)
clauses = clauses + survey(15,2, 2,4)
# clauses = clauses + survey(2,15, 2,3)



# g = Glucose3(with_proof=False)
# g = MapleCM(with_proof=False)
g = Glucose42(with_proof=False)
from pysat.process import Processor
processor = Processor(bootstrap_with=clauses)
processed = processor.process()
# print(processed.clauses)
# print(processed.status)
print(len(clauses),len(processed.clauses))
for clause in clauses:
    g.add_clause(clause)
# print('here')
print('solve',g.solve())

print('core',g.get_core())
print('accum_state',g.accum_stats())
print('proof',g.get_proof())
print('model',g.get_model())
for i in g.get_model():
    if i >0:
        print(i)
print(g.get_status())