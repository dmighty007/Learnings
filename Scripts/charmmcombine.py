import re
import numpy as np
class CHARMMCOMBINE:
    def __init__(self, base_file, add_file, comb_file):
        base = open(base_file).read()
        add1 = open(add_file).read()
        base_blocks = self.getBlocks(base)
        add_blocks = self.getBlocks(add1)
        base_atoms = self.getAtomTypes(base_blocks)
        add_atoms = self.getAtomTypes(add_blocks)
        overlapping_atoms = set(base_atoms).intersection(set(add_atoms))

        temp_atoms = base_atoms.copy()
        for overlap in overlapping_atoms:
            new_name = self.Generate_Random_Residue_Name(overlap, temp_atoms)
            print(f"change {overlap} to {new_name}")
            add1 = add1.replace(overlap, new_name)
            temp_atoms.append(new_name)
            
        add1_blocks = self.getBlocks(add1)

        add_heads = self.block_heads(add1_blocks)
        add_dict = self.dictMaker(add_heads, add1_blocks)
        base_heads = self.block_heads(base_blocks)
        base_dict = self.dictMaker(base_heads, base_blocks)

        new_dict = {}
        all_heads = ['[ defaults ]', '[ atomtypes ]', '[ nonbond_params ]', '[ bondtypes ]',
        '[ pairtypes ]', '[ angletypes ]', '0[ dihedraltypes ]', '1[ dihedraltypes ]']
        for head in all_heads:
            if 'defaults' not in head:
                if head in list(base_dict.keys()):
                    prv_list = base_dict[head]
                    if head in list(add_dict.keys()):
                        prv_list.extend(add_dict[head][1:])
                    new_dict[head] = prv_list 
                else:
                    new_dict[head] = add_dict[head]
            else:
                new_dict[head] =  base_dict[head]
                
                
        self.ff_string = "\n".join(["\n".join(list(new_dict.values())[i]) for i in range(len(new_dict.values()))])
        with open(comb_file, "w") as f:
            f.write(self.ff_string)

    block_heads = lambda self, add1_blocks : [block[0] for block in add1_blocks]

    def getBlocks(self, base):
        base = base.split("\n")
        index_list = []
        for i, line in enumerate(base):
            if line.startswith("["):
                index_list.append(i)
        index_list.append(len(base))        
        return [base[index_list[i]:index_list[i+1]] for i in range(len(index_list)-1)]
    
    def getAtomTypes(self, add_blocks):
        at_types = []
        for i in range(len(add_blocks)):
            if "atomtypes" in add_blocks[i][0]:
                for atom_line in add_blocks[i][1:]:
                    if not atom_line.startswith(";") and len(atom_line.strip()):
                        at_types.append(atom_line.split()[0].strip())
        return at_types

    def Generate_Random_Residue_Name(self, at, temp_atoms):
        string = re.findall("[a-zA-z]+", at)[0]
        number = re.findall("\d+", at)[0]
        len_res = len(at)
        len_num = len_res - len(string)
        while True:
            #if int(number) < 100:
            #    guess = np.random.randint(99)
            #if int(number)>100:
            #    guess = np.random.randint(100, 999)
            if len_num == 3:
                guess = np.random.randint(100, 999)
            elif len_num == 2:
                guess = np.random.randint(10, 99)
            elif len_num == 1:
                guess = np.random.randint(10)
            elif len_num == 4:
                guess = np.random.randint(1000, 9999)
            else:
                print(f"Error,{at}")
            newname = string + str(guess)
            if newname not in temp_atoms:
                break
        return newname
        
    def dictMaker(self, add_heads, add1_blocks):
        my_dict = {}
        count = 0
        for i, head in enumerate(add_heads):
            if "dihedraltypes" not in head:
                my_dict[head] = add1_blocks[i]
            else:
                my_dict[str(count)+head] = add1_blocks[i]
                count +=1
        return my_dict
