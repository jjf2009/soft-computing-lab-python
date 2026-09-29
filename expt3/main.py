import math

in1=float(input("Enter x1: "))
in2=float(input("Enter x2: "))

a11=float(input("Enter v11: "))
a12=float(input("Enter v12: "))
a21=float(input("Enter v21: "))
a22=float(input("Enter v22: "))

b1=float(input("Enter w1: "))
b2=float(input("Enter w2: "))

alpha=float(input("Enter learning rate: "))
target=float(input("Enter actual output: "))
tol=float(input("Enter error threshold: "))

def activation(net):
    return 1/(1+math.exp(-net))

def act_grad(y):
    return y*(1-y)

itr=0
err=float('inf')
while err > tol:
    itr += 1

    h1_in = in1*a11 + in2*a12
    h1 = activation(h1_in)

    h2_in = in1*a21 + in2*a22
    h2 = activation(h2_in)

    out_in = h1*b1 + h2*b2
    out = activation(out_in)

    err = 0.5*(target-out)**2

    grad_out = (target-out)*act_grad(out)

    grad_h1 = b1*grad_out*act_grad(h1)
    grad_h2 = b2*grad_out*act_grad(h2)

    b1 = b1 + alpha*grad_out*h1
    b2 = b2 + alpha*grad_out*h2

    a11 = a11 + alpha*grad_h1*in1
    a12 = a12 + alpha*grad_h1*in2
    a21 = a21 + alpha*grad_h2*in1
    a22 = a22 + alpha*grad_h2*in2

    print(f"\nEpoch {itr}")
    print(f"Output = {out:.6f}")
    print(f"Error  = {err:.6f}")

    print("Updated W:")
    print(f"w1 = {b1:.6f}, w2 = {b2:.6f}")

    print("Updated V:")
    print(f"v11 = {a11:.6f}, v12 = {a12:.6f}")
    print(f"v21 = {a21:.6f}, v22 = {a22:.6f}")
print("Training complete")
print("Epochs=",itr)
print("Final Output=",out)
print("Final Error=",err)
print("Updated W:",b1,b2)
print("Updated V:",a11,a12,a21,a22)
