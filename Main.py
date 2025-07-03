import sys
import random
import math
import time


sys.stdin = open('input.txt', 'r')

gates = dict()

wires = []


while True:
    try:
        inp = input()
        inp = inp.split()
        if inp[0] == 'wire':
            one = inp[1].split('.')
            two = inp[2].split('.')
            tup = (one[0],int(one[1][1:])-1,two[0],int(two[1][1:])-1)
            gates[one[0]]["connect"].append(tup)
            gates[two[0]]["connect"].append(tup)
            wires.append(tup)
            pass
        elif inp[0]=='pins':
            gate = inp[1]
            for i in range(2,len(inp),2):
                gates[gate]['pins'].append((int(inp[i]),int(inp[i+1])))
            pass
        else:
            gates[inp[0]] ={"width": int(inp[1]), "height": int(inp[2]), "pins": [],"connect":[]}
    except Exception as e:
        # print(e)
        break


class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.num_sets = n
        self.largest = 1

    def find(self, a):
        acopy = a
        while a != self.parent[a]:
            a = self.parent[a]
        while acopy != a:
            self.parent[acopy], acopy = a, self.parent[acopy]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a

            self.num_sets -= 1
            self.parent[b] = a
            self.size[a] += self.size[b]
            self.largest = max(self.largest,self.size[a])

    def set_size(self, a):
        return self.size[self.find(a)]

    def __len__(self):
        return self.num_sets

dsu = DisjointSetUnion(len(gates))
for i in wires:
    gate1=int(i[0][1:])-1
    gate2=int(i[2][1:])-1
    dsu.union(gate1,gate2)

allcomps = [None]*(len(gates))
for i in range(len(gates)):
    ind = dsu.find(i)
    if allcomps[ind] is None:
        allcomps[ind] = dict()
        allcomps[ind][f'g{i+1}'] = gates[f'g{i+1}']
    else:
        allcomps[ind][f'g{i+1}'] = gates[f'g{i+1}']

better_comps = []

for i in allcomps:
    if i is not None:
        better_comps.append(i)

tot = 0
for ab in better_comps:
    if len(ab)==1:
        num_iterations = 1
    elif len(ab)<25:
        num_iterations = 10**5
    else:
        num_iterations = int((10**5 )*(((len(ab)-25)**2)*(9/(975*975)))) + 10**5
    runs= min(10,(10 **3)//len(ab))
    tot+=runs*num_iterations

class ProgressBar:
    def __init__(self, total_iterations, bar_length=60, update_interval=1):
        self.total_iterations = total_iterations  
        self.bar_length = bar_length  
        self.current_iteration = 0  
        self.start_time = time.time() 
        self.last_update_time = time.time() 
        self.update_interval = update_interval 

    def update(self):

        self.current_iteration += 1
        current_time = time.time()
        
        # Only update the display if enough time has passed since the last update
        if current_time - self.last_update_time >= self.update_interval or self.current_iteration == self.total_iterations:
            self.display()
            self.last_update_time = current_time

    def display(self):

        progress = self.current_iteration / self.total_iterations
        block = int(self.bar_length * progress)
        bar = '█' * block + '-' * (self.bar_length - block)


        elapsed_time = time.time() - self.start_time
        if progress > 0:
            estimated_total_time = elapsed_time / progress
            estimated_remaining_time = estimated_total_time - elapsed_time
        else:
            estimated_remaining_time = 0


        elapsed_time_str = self._format_time(elapsed_time)
        remaining_time_str = self._format_time(estimated_remaining_time)


        sys.stdout.write(f'\r[{bar}] {self.current_iteration}/{self.total_iterations} ({progress:.2%}) '
                         f'Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str}')
        sys.stdout.flush()


        if self.current_iteration == self.total_iterations:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def _format_time(self, seconds):
        """Formats time in seconds into HH:MM:SS."""
        hrs, rem = divmod(seconds, 3600)
        mins, secs = divmod(rem, 60)
        return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"


progress_bar = ProgressBar(tot)

def onecomp(gates):


    maxw = 0
    maxh = 0
    totarea = 0
    for i in gates:
        totarea += gates[i]['width']*gates[i]['height']
        maxw = max(maxw,gates[i]['width'])
        maxh = max(maxh,gates[i]['height'])

    blocks = (math.ceil(math.sqrt(len(gates))))**2
    peff = totarea/(blocks*maxw*maxh*2)
    factor = (peff**2) +1
    maxw*=factor
    maxh*=factor

    maxw= int(maxw)
    maxh = int(maxh)


    def manhattan_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)


    def len_init(placement, gates):
        total_length = 0
        
        for gate in gates:
            total_length+=single_len(placement,gate,gates)

        
        return total_length//2

    def single_len(placement,gate,gates):
        total_length = 0
        wires = gates[gate]["connect"]
        for wire in wires:
            gate1, pin1_idx, gate2, pin2_idx = wire
            g1_x, g1_y = placement[gate1]
            g2_x, g2_y = placement[gate2]
            
            p1_x, p1_y = gates[gate1]["pins"][pin1_idx]
            p2_x, p2_y = gates[gate2]["pins"][pin2_idx]
            
            pin1_abs_x, pin1_abs_y = g1_x + p1_x, g1_y + p1_y
            pin2_abs_x, pin2_abs_y = g2_x + p2_x, g2_y + p2_y

            total_length += manhattan_distance(pin1_abs_x, pin1_abs_y, pin2_abs_x, pin2_abs_y)
            
        
        return total_length
            
    def is_overlapping(gate1, gate2, placement):
        x1, y1 = placement[gate1]
        x2, y2 = placement[gate2]
        w1, h1 = gates[gate1]["width"], gates[gate1]["height"]
        w2, h2 = gates[gate2]["width"], gates[gate2]["height"]
        
        
        if x1<x2:
            if y1<y2:
                return x1+w1>x2 and y1+h1>y2
            else:
                return x1+w1>x2 and y2+h2>y1
        else:
            if y1<y2:
                return x2+w2>x1 and y1+h1>y2
            else:
                return x2+w2>x1 and y2+h2>y1



    def has_overlap(placement, gate):
        for i in placement:
            if gate==i:
                continue
            else:
                if is_overlapping(i,gate,placement):
                    return True
        return False

            

    def simulated_annealing(gates, initial_temperature, cooling_rate, num_iterations):


        n = len(gates)
        m = math.ceil(math.sqrt(n))
        perm = [i for i in range(n)]
        perm.sort(key = lambda x:random.random())

        placement = dict()
        # Intial Random Placement of gates

        j=0
        for i in gates:
            ind =  perm[j]
            j+=1
            h = ind//m
            w = ind%m
            placement[i] = (w*maxw,h*maxh)

        
        
        
        current_length = len_init(placement, gates)
        best_placement = placement.copy()
        best_length = current_length
        temperature = initial_temperature
        
        for _ in range(num_iterations):
            progress_bar.update() 

            # Perturbation: randomly move one gate to a nearby position
            gate_to_move = random.choice(list(gates.keys()))
            old_position = placement[gate_to_move]
            new_position = (old_position[0] + random.choice([-2,-1,0,1,2]),
                            old_position[1] + random.choice([-2,-1,0,1,2]))
            
            if old_position==new_position:
                continue

            # Update placement
            temp = single_len(placement,gate_to_move,gates)
            placement[gate_to_move] = new_position
            temp2 = single_len(placement,gate_to_move,gates)

            # Ensure no overlaps
            if has_overlap(placement, gate_to_move):
                placement[gate_to_move] = old_position
                continue
            
            new_length = current_length - temp + temp2
            delta_length = new_length - current_length
            
            # Decide whether to accept the new placement
            if delta_length < 0 or random.uniform(0,1) < math.exp(-delta_length / temperature):
                current_length = new_length
                if current_length < best_length:
                    best_length = current_length
                    best_placement = placement.copy()
            else:
                placement[gate_to_move] = old_position
            
            # Cool down the temperature
            temperature *= cooling_rate
        
        return best_placement, best_length

    # Setting Parameters for Simulated Annealing
    initial_temperature = 1000
    cooling_rate = 0.999
    if len(gates)==1:
        num_iterations = 1
    elif len(gates)<25:
        num_iterations = 10**5
    else:
        num_iterations = int((10**5 )*(((len(gates)-25)**2)*(9/(975*975)))) + 10**5

    # Run Simulated Annealing
    best_placement = None
    best_length = float('inf')
    runs= min(10,(10 **3)//len(gates))
    
    for i in range(runs):
        a, b = simulated_annealing(gates, initial_temperature, cooling_rate, num_iterations)
        if b<best_length:
            best_placement=a.copy()
            best_length=b

    minx=float('inf')
    miny=float('inf')
    maxx=float('-inf')
    maxy=float('-inf')
    # print(best_placement)
    for gate in best_placement:
        x,y = best_placement[gate]
        maxx = max(maxx,x+gates[gate]['width'])
        minx = min(minx,x)
        maxy = max(maxy,y+gates[gate]['height'])
        miny = min(miny,y)
    # print(maxx,minx,maxy,miny)

    # Return the results
    return ((maxx-minx,maxy-miny),best_placement,best_length,(minx,miny))

bbox_w = 0
bbox_h = 0
best_placement = dict()
best_lenght = 0
for comp in better_comps:
    bbox,placement,wirelen,mintup = onecomp(comp)
    best_lenght+=wirelen
    for block in placement:
        best_placement[block]=(bbox_w+placement[block][0]-mintup[0],placement[block][1]-mintup[1])
    bbox_w+=bbox[0]
    bbox_h = max(bbox_h,bbox[1])


sys.stdout = open("output.txt",'w')
print(f'bounding_box {bbox_w} {bbox_h}')
for gate in best_placement:
    print(gate,best_placement[gate][0],best_placement[gate][1])
print("wire_length", best_lenght)


