import torch as tc
import numpy as np
from mpi4py import MPI

def get_comm():
    comm = MPI.COMM_WORLD
    return comm

@tc.no_grad()
def sync_state(model, optimizer, scheduler, comm, root):
    model_state_dict = comm.bcast(model.state_dict(), root=root)
    optimizer_state_dict = comm.bcast(optimizer.state_dict(), root=root)
    if scheduler is not None:
        scheduler_state_dict = comm.bcast(scheduler.state_dict(), root=root)

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state_dict)

@tc.no_grad()
def sync_grads(model, comm):
    for p in model.parameters():
        p_grad_local = p.grad.numpy()
        p_grad_global = np.zeros_like(p_grad_local)
        comm.Allreduce(sendbuf=p_grad_local, recvbuf=p_grad_global, op=MPI.SUM)
        p.grad.copy_(tc.FloatTensor(p_grad_global) / comm.Get_size())
