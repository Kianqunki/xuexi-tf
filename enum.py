def gen_batch(batch_size, num_steps):
    for i in range(batch_size):
        yield (i, num_steps)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(2, num_steps)

for i,epoch in enumerate(gen_epochs(2,5)):
    print i,epoch
    for step, (X,Y) in enumerate(epoch):
        print step,X,Y
