import random
import sys
import getopt

if len(sys.argv)!=15:
    print "usage: flup.py -n <Grid size> -u <Utility Function> -t <number of tasks> -c <compCap> -i <insight> -l <leakage> -o <output_file>"
    sys.exit()
    
opts, args = getopt.getopt(sys.argv[1:], "n:u:t:c:i:l:o:", ["grid_size=", "utility=", "Tasks=", "compCap=", "insight=", "output="])

for o,a in opts:
    if o == "-n":
        n = int(a)
    if o == "-u":
        choice_of_u = int(a)
    if o == "-t":
        nTasks = int(a)
    if o == "-c":
        compCap = int(a)
    if o == "-i":
        insight = float(a)
    if o == "-o":
        output_file = a
    if o == "-l":
        leakage = float(a)

pop_size = 100
mutation_prob = 0.2
crossover_prob = 0.10
number_of_gen = 3000
taskList=[] #List of allowed tasks

for i in range(0, nTasks):
  taskList.append(i) #Makes a list of all tasks [0,1,2 .... nTasks-1]

#All instructions which are not pulling arms are negative

def TaskToInfo(task): # How does task become info
    c = 3
    k = 5                 #Function variables for TaskToInfo 
    if task != 0:
        info = c*task + k     #Define the task no. to information conversion here
    else:
        info = 0
    return info

class Player:
  def __init__(self, pos, genome):
      self.p_pos = pos      #x,y position
      self.p_genome = genome        #Genome sequencea
      self.p_info = []          #information about tasks available to the player
      self.p_received =[]
      for i in range(0,nTasks):
        self.p_info.append(0)

def PosToPlayer(targetpos, PlayerList): #Given a position, returns the player at that position
  for i in range(len(PlayerList)-1):
    if PlayerList[i].p_pos==targetpos:
      return i

def mypos(self):                #Test function for Player
  print("My pos is " + self.p_pos)
  
def GenomeToInfo(player):       #Converts entire genome of tasks to info
  transcript = player.p_genome
  for i in range(0,len(transcript)):
      if transcript[i] in taskList: 
            player.p_info[transcript[i]]+=(TaskToInfo(transcript[i]))
  return(player.p_info)

def mode(list):
    mode=max(set(list), key=list.count) #Finds out most performed task
    return mode

def Initialize_Received(Player):
    for i in range(0,nTasks):
        Player.p_received.append(0)
    return Player.p_received

def SendToNeighbor(player, PlayerList, task): 
  info_sent = leakage * player.p_info[task]		#Sends 'info_sent' amount of information about task 'task' to neighbour 'player'. Iterates over 8 nearest neighbors. 
  info_quanta = float((info_sent/8))
  pos = player.p_pos
  for i in [pos[0]-1, pos[0], pos[0]+1]:
    x = i # x denotes x coordinate of neighbour
    if i == -1:
      x = n-1 # Player in left column's left neighbour is the right column player
    if i == n:
      x = 0 # Player in right column's right neighbour is the left column player
    for j in [pos[1]-1, pos[1],pos[1]+1]:
      
      y = j
      if j == -1:
        y = n-1 # Player in left column's left neighbour is the right column player
      if j == n:
        y = 0 # Player in right column's right neighbour is the left column player

      if [x,y] != [pos[0],pos[1]]: #Wouldn't want me sending info to me
        current_neighbour = PlayerList[PosToPlayer([x,y], PlayerList)] # Identifies neighbour 
        current_neighbour.p_received[task] += info_quanta
        player.p_info[task] -= info_quanta

def diff_index(PlayerList): #Differentiation index
  List_of_counts = [] #Creates a list of lists with count of each task of each player
  sum_of_counts = [] #List of number of times task i has been done by the n*n bandits
  ratio_of_max = [] #Ratio of mode to sum_of_counts for that task
  for i in range(0,len(PlayerList)):
    count =[] #Creates an empty list
    for j in range(0, nTasks):
      count.append(0) #Sets count of all tasks to zero
      sum_of_counts.append(0)
    ratio_of_max.append(0)
    List_of_counts.append(count)
    

  for i in range(0,len(PlayerList)-1):
    for j in PlayerList[i].p_genome:
      if j > 0:
        List_of_counts[i][j] += 1 #If a task has been done(sharing is not counted), increments by 1
        sum_of_counts[j] += 1 
  output = [0,0,0]  
                     #initialize specialization index
  max_si_net = 0.0
  for i in range(0,len(PlayerList)-1):
    max_si = 0.0
    max_c = 0   #Looks for the mode
    for j in range(0, nTasks):
      if List_of_counts[i][j] > max_c: #If it is not the mode, its si does not matter
        max_c = List_of_counts[i][j] 
        if sum_of_counts[j] > 0:
          max_si = float(float(List_of_counts[i][j])/sum_of_counts[j])
          player = i
          task = j
    else:
        if List_of_counts[i][j] == max_c:
          if sum_of_counts[j] > 0:
            if (float(float(List_of_counts[i][j]))/sum_of_counts[j]) > max_si: #If two tasks are both mode, the more specialised one is chosen
              max_si = float(float(List_of_counts[i][j])/sum_of_counts[j])
              player = i
              task = j
              
    if max_si > max_si_net:
        max_si_net = max_si
        output[0] = player
        output[1] = task
        output[2] = max_si
          
  return output

def temperature(i, runs, cooling_schedule, start_temp, end_temp):
    i = float(i)
    T0 = float(start_temp)
    Tn = float(end_temp)
    N = float(runs)
    
    if cooling_schedule == 0:
        Ti = T0 -i*(T0-Tn)/N
    
    if cooling_schedule == 1:
        Ti = T0*(Tn/T0)**(i/N)
    
    if cooling_schedule == 2:
        A = (T0-Tn)*float(N+1)/N
        B = T0 -A
        Ti = A/(i+1) +B
    
    if cooling_schedule == 3:
        print "warning: cooling_schedule '3' does not work as described"
        print "switching to cooling_schedule '5'"
        cooling_schedule = 5
    
    if cooling_schedule == 4:
        Ti = (T0-Tn)/(1+math.exp(0.01*(i-N/2))) +Tn;
    
    if cooling_schedule == 5:
        Ti = 0.5*(T0 -Tn)*(1+math.cos(i*math.pi/N)) +Tn
    
    if cooling_schedule == 6:
        Ti = 0.5*(T0-Tn)*(1-math.tanh(i*10/N-5)) +Tn;
    
    if cooling_schedule == 7:
        Ti = (T0-Tn)/math.cosh(i*10/N) +Tn;
    
    if cooling_schedule == 8:
        A = (1/N)*math.log(T0/Tn)
        Ti = T0*math.exp(-A*i)

    if cooling_schedule == 9:
        A = (1/N**2)*math.log(T0/Tn)
        Ti = T0*math.exp(-A*i**2);
    
    return Ti


def Population_Generator():  #pop_size is the population size over which we run optimization. Higher it is, better the optimization
    population=[]
    for i in range(0,pop_size):
        PlayerList = []
        for j in range(0, n):				# Each member of the population has n*n sub-members as we are trying to optimize social welfare
            for k in range(0,n):    
                sub_genome = Genome_Generator()         #Generates genome for a sub-member
                p1 = Player([j,k],sub_genome)           #Creates a player at [j,k] with generated genome
                p1.p_info=GenomeToInfo(p1)              #Adds info to that player
                p1.p_received=Initialize_Received(p1)
                PlayerList.append(p1)               
        PlayerList.append(0)    							#Assigns fitness to zero for now
        population.append(PlayerList)                   #Adds the n*n sub-member as one PlayerList in our Genetic Algorithm population
    return population						#Returns a list of pop_size members, each with n*n lists of their own
  
def Genome_Generator():
    v_genome = [] 					#v_genome is virtual genome
    for i in range(0,compCap):
        task = random.randint(0, (nTasks-1))
        v_genome.append(task)    
    return v_genome						#Returns a genome with no cooperation for one sub-member

def Sort_By_Fitness(population): 
    for i in range(0,len(population)):
        population[i][n*n] = 0
    for j in range(0,len(population)):
        population[j][n*n] = SocialWelfare(population[j])	#Assigns population[j][n*n] as the fitness of member j of the population 
    population.sort(key = lambda x: int (x[n*n]), reverse = True)       #Sorts by fitness, fittest is index 0

    return population
  
def Crossover_of_genome(PlayerList1, PlayerList2):
    genome1=[]
    genome2=[]
    for i in range(0,len(PlayerList1)-1):
        for j in range(0,len(PlayerList1[i].p_genome)):
            genome1.append(PlayerList1[i].p_genome[j])      #Creates one string which has compCap*n*n tasks (of the PlayerLIst) in it. This is what we optimize.
    for i in range(0,len(PlayerList2)-1):
        for j in range(0,len(PlayerList2[i].p_genome)):
            genome2.append(PlayerList2[i].p_genome[j])      #Creates compCap*n*n long string for player 2.
            
    number_of_cuts = random.randrange(1,4) 		#Chooses anywhere between 1 to 4 crossover episodes
    cut = 0
    cut_copy = cut	 #Just a variable to store the cut location
    offspring = [] #The offspring of the two genomes
    cross_prob = random.random()
    if cross_prob < crossover_prob: #Checks if there should be no crossover
        parent_prob=random.random() #Randomly picks one of the two parents
        if parent_prob < 0.5:
            for j in range(0,compCap*n*n):
                x = random.randrange(0,100)     #Generates a random number                            
                if x < mutation_prob*1000:      #Checks is mutation should be made
                    offspring.append(random.randint(-(nTasks-1),nTasks-1))
                else:               #Randomly decides to append from either genome 1 or 2
                    offspring.append(genome1[j])
        else:
            for j in range(0,compCap*n*n):
                x = random.randrange(0,100)     #Generates a random number                            
                if x < mutation_prob*1000:      #Checks is mutation should be made
                    offspring.append(random.randint(-(nTasks-1),nTasks-1))
                else:               #Randomly decides to append from either genome 1 or 2
                    offspring.append(genome2[j])
    else:        
        for i in range(0,number_of_cuts):
            cut = random.randrange(cut_copy,(compCap*n*n)) #Creates a cut where crossover will take place
            for j in range(cut_copy,cut):
                x = random.randrange(0,100)     #Generates a random number
                if x < mutation_prob*1000:      #Checks is mutation should be made
                    offspring.append(random.randint(-(nTasks-1),nTasks-1))
                else:               #Randomly decides to append from either genome 1 or 2
                    if x >= mutation_prob*1000+1 and x < 1000*mutation_prob-500*(1-mutation_prob):
                        offspring.append(genome1[j])
                    else:
                        offspring.append(genome2[j])                        
            cut_copy=cut
            if i==number_of_cuts-1 and len(offspring)!=n*n*compCap: #Fills up remaining space in case the cuts never reach n*n*compCap
                for k in range(len(offspring),n*n*compCap):
                    x = random.randrange(0,1000)
                    if x < mutation_prob*1000:
                        offspring.append(random.randint(-(nTasks-1),nTasks-1))
                    else:
                        if x >= mutation_prob*1000+1 and x < 1000*mutation_prob-500*(1-mutation_prob):
                            offspring.append(genome1[j])
                        else:
                            offspring.append(genome2[j])                        
    offspring.append(0)
    child=[]
    l=0
    for i in range(0,n):        #Converts the compCap*n*n long string into n*n players with compCap long genomes
        for j in range(0,n):                           
            member=[]
            for k in range(0,compCap):
                member.append(offspring[l])
                l+=1
            p1=Player([i,j],member)
            p1.p_info=GenomeToInfo(p1)
            p1.p_received=Initialize_Received(p1)
            child.append(p1)
    child.append(0)
    return child

def New_Generation(population):
    if pop_size % 2 == 0:
        Median_Fitness = float((population[(pop_size/2)][n*n] + population[(pop_size/2)-1][n*n])/2) #Calculates Median Fitness
    else:
        Median_Fitness = float(population[(pop_size)/2][n*n])
    prob_of_index = []      
    Sum_of_deviations = 0
    for i in range(0,len(population)):
        if (population[i][n*n] - Median_Fitness) > 0:
            Sum_of_deviations += population[i][n*n] - Median_Fitness    #Sums up all positive deviations from Median_Fitness
    for i in range(0,len(population)):
        if (population[i][n*n] - Median_Fitness) > 0:
            prob_of_rep = float((population[i][n*n] - Median_Fitness)/Sum_of_deviations)    #Prob. of a player reproducing is proportional to his deviation from Median_Fitness
        else:
            prob_of_rep = 0
        prob_of_index.append(prob_of_rep)
    new_pop = []
    for i in range(0,len(population)):
        parent=[]
        for j in range(0,2):        #Chooses two parents based on probability
            r = random.random()
            index = 0
            while(r >= 0 and index < len(prob_of_index)):
              r -= prob_of_index[index]
              index += 1
            parent.append(index - 1)
        new_pop.append(Crossover_of_genome(population[parent[0]],population[parent[1]]))
    return new_pop

def SocialWelfare(PlayerList): #Returns social welfare = sum of utilities of all players
    sw = 0
    ExecuteSharing(PlayerList)
    for i in range(0,len(PlayerList)-1):
        sw += IndividualUtility(PlayerList[i]) #Adds utility of each player's information vector to social welfare.
    return sw
    
def IndividualUtility(Player): #Returns Individual utility of given player calculated from his information vector
    if choice_of_u == 0:
        return UtilityFunction0(Player.p_info)
    if choice_of_u == 1:
        return UtilityFunction1(Player.p_info)
    if choice_of_u == 2:
        return UtilityFunction2(Player.p_info)
    if choice_of_u == 3:
        return UtilityFunction3(Player.p_info)

def ispresent(i,info_vector): #Returns 1 if task no. i has non-zero information in info_vector, returns 0 otherwise
  if info_vector[i]>0:
    return 1
  else:
    return 0        
                     
def UtilityFunction0(info_vector):
  utility = 0
  for i in range(0,nTasks):
    utility+= info_vector[i]
  return utility

def UtilityFunction1(info_vector): 
  # A hierarchical utility function that awards more utility for higher tasks if information about lower tasks is present. The increase in utility is
  # not dependent on the magnitude of information about lower tasks, but just whether the information is present or not.
  # c is some constant that can be considered the "insight" the player has about a task given he knows how tasks lower than this work.
  # c < 1
  # Task  |  Utility contribution
  #   0   |  info_vector[0]
  #   1   |  info_vector[1] + c * [ispresent(0,info_vector)]
  #   2   |  info_vector[2] + c * [ispresent(1,info_vector) + ispresent(0,info_vector)]
  #   3   |  info_vector[3] + c * [ispresent(2,info_vector) + ispresent(1,info_vector) + is_present(0,info_vector)]
  utility = 0
  for i in range(0,nTasks):
    utility_of_i = info_vector[i]
    if info_vector[i] > 0:
        for j in range(0,i):
          utility_of_i += insight * ispresent(j,info_vector)      
    utility += utility_of_i
  return utility


def UtilityFunction2(info_vector):
  # A hierarchical utility function that awards more utility for higher tasks depending on the amount of information about lower tasks present.
  # The extra increase in utility is dependent on the magnitude of information present about lower tasks.
  # c is some constant that can be considered the "insight" the player has about a task given he knows what tasks are lower than this one
  # and what information reward they give.
  # c < 1
  # Task  |  Utility contribution
  #   0   |  info_vector[0]
  #   1   |  info_vector[1] + c * (info_vector[0])
  #   2   |  info_vector[2] + c * (info_vector[1] + info_vector[0])
  #   3   |  info_vector[3] + c * (info_vector[2] + info_vector[1] + info_vector[0])
    utility = 0
    for i in range(0,nTasks):
        utility_of_i = info_vector[i]
        if info_vector[i] > 0:
            for j in range(0,i):
                utility_of_i += insight * info_vector[j]
        utility += utility_of_i
    return utility

def UtilityFunction3(info_vector):
  utility = 0
  for i in range(0,nTasks):
    utility_of_i = info_vector[i]
    if info_vector[i] > 0:
        for j in range(0,i):
            if info_vector[j] > 0:
                for k in range(0,j+1):
                  utility_of_i += float(pow(insight,i-k)) * info_vector[k]      
    utility += utility_of_i
  return utility

def IndividualCooperativity(Player):
  # Number of Send Tasks in an organism's genome divided by the total genome length.
  n_send = 0.0
  for i in range(0,compCap):
    if Player.p_genome[i]<0:
      n_send += 1
  
  return float(n_send/float(compCap))
  
def GlobalCooperativity(PlayerList):
  gc = 0
  for i in range(0,len(PlayerList)-1):
    gc += IndividualCooperativity(PlayerList[i])
  
  return gc/len(PlayerList)

def SpecialisationIndex(PlayerList):
    si = 0.0
    for i in range(0,len(PlayerList)-1):
            genome=PlayerList[i].p_genome
            noofoccur=genome.count(mode(genome))
            si+=float(noofoccur/float(compCap*n*n))

    return si

def ExecuteSharing(PlayerList):
    for i in range(0,len(PlayerList)-1): #For each player in playerlist
        transcript = PlayerList[i].p_genome
        for j in range(0,compCap): #Check each letter in player genome  
            if transcript[j]<0: #If letter is a negative number
                tasktosend= -1 * transcript[j] #Convert to positive number
                SendToNeighbor(PlayerList[i], PlayerList, tasktosend) #Send this task to neighbor
    for i in range(0,len(PlayerList)-1):
        for j in range(0,nTasks):            
            PlayerList[i].p_info[j] += PlayerList[i].p_received[j] #Sums his received and existing info
            PlayerList[i].p_received[j] = 0 #Makes his received zero for the next time step

            
def sort_a_string(PlayerList):
    for i in range(0,len(PlayerList)-1):
        x = []
        for j in range(0,len(PlayerList[i].p_genome)):
            x.append(PlayerList[i].p_genome[j])
        x.sort()
        PlayerList[i].p_genome = x
    return PlayerList

f = open(output_file, "w")


pop=Population_Generator()
##for i in range(0,len(pop)):
##    for j in range(0,len(pop[i])-1):
##        print pop[i][j].p_info
#print "----------------"        
pop=Sort_By_Fitness(pop)
##for i in range(0,len(pop)):
##    print pop[i][n*n]
best_pop = pop[0]
for i in range(0,len(best_pop)-1):
    print best_pop[i].p_genome
    f.write("#"+str(best_pop[i].p_genome)+"\n")
print best_pop[n*n]
print "-----------------------------------------------------------"
gc = GlobalCooperativity(best_pop)
print gc
for k in range(0,number_of_gen):    
    New = New_Generation(pop)
    New = Sort_By_Fitness(New)
    if New[0][n*n] > best_pop[n*n]:
        best_pop = New[0]
    pop = New
    gc = GlobalCooperativity(best_pop)
    sw = SocialWelfare(best_pop)
    print gc
    f.write(str(k)+"  "+str(sw)+"  "+str(gc)+"\n")
best_pop = sort_a_string(best_pop)
##    for i in range(0,len(best_pop)-1):
##        print best_pop[i].p_genome
##    print "-----------------------------------------------------------"
for i in range(0,len(best_pop)-1):
    print best_pop[i].p_genome
    f.write("#"+str(best_pop[i].p_genome)+"\n")
print best_pop[n*n]
print "#"+"Specialization index= "+str(diff_index(best_pop))+"\n"
f.write("#"+"Specialization index= "+str(diff_index(best_pop))+"\n")
f.close()
#print "\n",off
