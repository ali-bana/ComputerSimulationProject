import sys
from RandomVariableGenerator import exponentialGenerator
from functools import cmp_to_key
import numpy as np

class Event:
    def __init__(self, kind, time, description):
        self.time = time
        self.kind = kind
        self.description = description

    def __str__(self):
        return '|event: ' + self.kind + ', ' + 'time: ' + str(self.time) + ' ' +self.description + "|"
    def __repr__(self):
        return '|event: ' + self.kind + ', ' + 'time: ' + str(self.time) + ' ' +self.description +"|"

class Task:
    def __init__(self, type, deadline, arrival_time):
        self.type = type
        self.deadline = deadline
        self.arrival_time = arrival_time

    def __repr__(self):
        return '|task: ' + str(self.type) + ', arr_time: ' + str(self.arrival_time) + ', deadline: ' +str(self.deadline) + "|"


def task_compare(o1, o2):
    """
used to sort the tasks
    :param o1:
    :param o2:
    :return:
    """
    if type(o2) != type(o1):
        return -1
    if o1.type == o2.type:
        if o1.arrival_time < o2.arrival_time:
            return -1
        return 1
    if o1.type == 1:
        return -1
    return 1


def next_task_time(time, task_l):
    """
    :param time: current time of the system
    :param task_l: the lambda of next task interval
    :return: time to expect a new task
    """
    return time + exponentialGenerator(task_l)


def task_generator(task_l, deadline_mean, time):
    """
    :param task_l: is the lambda of arrival intervals
    :param deadline_l: is the mean of deadlines
    :param time: is the current time of the system
    :returns: a task to be added to system and time for the next task
    """
    deadline = exponentialGenerator(1 / deadline_mean)

    if np.random.uniform(0, 1) < 0.1: # to make type 1 task with probability of 0.1
        task = Task(1, deadline+time, time)
    else:
        task = Task(2, deadline+time, time)

    return task, next_task_time(time, task_l)


def input_parser(path):
    """
    :param path: path to the input file
    :return: needed parameters to run the simulation
    """
    f = open(path)
    temp = []
    for l in f:
        temp.append(l)

    server_num = int(temp[0].split()[0])
    arrival_landa = float(temp[0].split()[1])
    deadline_mean = float(temp[0].split()[2])
    timer_rate = float(temp[0].split()[3])

    cores = []
    for i in range(1, len(temp)):
        t = []
        splt = temp[i].split()
        m = int(splt[0])
        for j in range(1, len(splt)):
            t.append(float(splt[j]))
        cores.append(t)
    return server_num, arrival_landa, deadline_mean, timer_rate, cores


def deadline_remover(server_Q, scheduler_Q, task_id):
    """
used to remove a task when it's deadline has come
    :param server_Q: the Q of all servers
    :param scheduler_Q: th Q the scheduler
    :param task_id: the id of the task
    """
    for i in range(len(scheduler_Q)):
        if str(id(scheduler_Q[i])) == task_id:
            del scheduler_Q[i]
            return
    for j in range(len(server_Q)):
        for i in range(len(server_Q[j])):
            if str(id(server_Q[j][i])) == task_id:
                del server_Q[j][i]
                return

def schedule(sch_Q, server_Q, cores, cores_lambda, fel, time):
    """
removes one task from the scheduler Q and then add it to the server_Q by the proposed priority
    :param sch_Q: the scheduler Q
    :param server_Q: the server Q
    """
    if len(sch_Q) == 0:
        return
    min = 100000000
    index = []
    for i in range(len(server_Q)):
        if len(server_Q[i]) == min:
            index.append(i)
        elif len(server_Q[i]) < min:
            min = len(server_Q[i])
            index = [i]

    i = np.random.randint(0, len(index))
    task = sch_Q.pop(0)
    server_Q[index[i]].append(task)
    server_Q[index[i]].sort(key=cmp_to_key(task_compare))
    start_task(server_Q, cores, cores_lambda, index[i], fel, time)


def start_task(server_Q, cores, cores_lambda, server, fel, time):
    """
to notify a server that a new task is added to it's Q or a task has been departed. Checks the input Q and
cores, then starts a task if Q is not empty and there exists an empty core
    :param server_Q: 2D array of server Qs
    :param cores: 2D array of server cores
    :param cores_lambda: 2D array of parameter of the server cores
    :param server: the server that should be notified
    :param fel:
    :param time:
    """
    if len(server_Q[server]) == 0:
        return
    for core in range(len(cores[server])):
        if cores[server][core] == -1:  # if there is an empty core
            task = server_Q[server].pop(0)
            cores[server][core] = task
            service_time = exponentialGenerator(cores_lambda[server][core])
            fel.append(Event('dep', time+service_time, str(server) + ' ' + str(core)))  # add departure event to FEL
            return

def departure(dep, server_Q, cores, cores_lambda, time, fel):
    """
when departure event call this function. it frees the core and call start_task
    :param dep: the descreption of event. format is 'server core'
    :param server_Q: 2D Q of all servers
    :param cores: 2D cores of all servers
    :param cores_lambda: 2D lambda of all cores
    :param time: time of the system
    :param fel: future event list of system
    """
    server = int(dep.split()[0])
    core = int(dep.split()[1])
    cores[server][core] = -1
    start_task(server_Q, cores, cores_lambda, server, fel, time)


def simulate(server_num, arrival_l, dead_m, sc_r, cores_lambda):
    """

    :param ser_num: total number of servers
    :param arr_l: the arrival times lambda
    :param dead_m: the mead deadline for each task
    :param sc_r: scheduler rate
    :param cores: 2D array of cores. Each row is the set of all cores of a server
    """
    fel = []  # the future event list
    server_Q = [[] for _ in range(server_num)]  # 2D array. each row is Q for one server
    schdle_Q = []  # Q of the scheduler
    time = 0  # time of the system
    cores = [[-1 for j in range(len(cores_lambda[i]))] for i in range(len(cores_lambda))]
    # makes a 2D array for cores, tasks place in cores, -1 indicates empty state
    fel.append(Event('END', 50000, ''))
    fel.append(Event('arrival', next_task_time(time, arrival_l), ''))
    fel.append(Event('sc_s', exponentialGenerator(sc_r) + time, ''))  # the time scheduler works


    while (True):
        if time >= 5000:
            return
        fel.sort(key=lambda x: x.time)
        e = fel.pop(0)  # pop the first event in the list
        time = e.time

        print('........................')
        print(e.kind, e.description)
        print('scheduler_q: ', len(schdle_Q))
        print('serverQ: ', [len(_) for _ in server_Q])
        print('cores', cores)
        print('--------------------------')



        if e.kind == 'arrival':
            # a new customer arrives
            task, next_arrival = task_generator(arrival_l, dead_m, time)
            schdle_Q.append(task)  # adding the arrived task to the scheduler Q
            fel.append(Event('arrival', next_arrival, ''))  # generating a future event for the next arrival
            schdle_Q.sort(key=cmp_to_key(task_compare))
            fel.append(Event('deadline', task.deadline, str(id(task))))
            pass
        elif e.kind == 'sc_s':
            fel.append(Event('sc_s', exponentialGenerator(sc_r) + time, ''))
            schedule(schdle_Q, server_Q, cores, cores_lambda, fel, time)  # schedule a task if the Q is not empty
            # scheduler should work here
            pass
        elif e.kind == 'dep':
            # a task is done
            departure(e.description, server_Q, cores, cores_lambda, time, fel)
            pass
        elif e.kind == 'deadline':
            # a task deadline has come
            deadline_remover(server_Q, schdle_Q, e.description)
            pass
        elif e.kind == 'END':
            # this is the end of simualtion
            return





if __name__ == '__main__':
    # path = sys.argv[1]
    path = 'a.txt'
    server_num, arrival_l, deadline_m, scheduler_rate, cores = input_parser(path)

    simulate(server_num, arrival_l, deadline_m, scheduler_rate, cores)