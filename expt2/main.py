import numpy as np

def get_vector_input(prompt):
    """Helper to get a list of numbers from the user and convert to a numpy column vector."""
    user_input = input(prompt)
    values = [float(val) for val in user_input.split()]
    return np.array(values).reshape(-1, 1)

def sgn(net):
    """Sign activation function: returns 1 if net >= 0, else -1."""
    return 1 if net >= 0 else -1

def main():
    print("--- Perceptron Learning Rule Setup ---")
    
    w = get_vector_input("Enter initial weights w1: ")
    
    c = float(input("Enter learning rate c: "))
    
    n = int(input("How many input patterns (x) are there: "))
    
    X_list = []
    d_list = []
    
    for i in range(n):
        x = get_vector_input(f"Enter x{i+1}: ")
        d = int(input(f"Enter target d{i+1}: "))
        X_list.append(x)
        d_list.append(d)
        
    print("\n---Perceptron Learning Rule---")
    
    for i in range(n):
        print(f"\n({i+1}) calculate net{i+1} = w{i+1}^T x{i+1}")
        x = X_list[i]
        d = d_list[i]
        
        net = np.dot(w.T, x)[0][0]
        print(f"w{i+1}^T = {np.round(w.T[0], 2)}")
        print(f"x{i+1} = \n{x}")
        print(f"net{i+1} = {net:.2f}")
        
        y = sgn(net)
        print(f"sgn(net{i+1}) = sgn({net:.2f}) = {y}")
        print(f"d{i+1} = {d}")
        
        if y == d:
            print("since both same nothing changes (w not changed)")
        else:
            if d == 1 and y == -1:
                print(f"Delta w{i+1} = +2cx")
                delta_w = 2 * c * x
                sign_char = "+"
            elif d == -1 and y == 1:
                print(f"Delta w{i+1} = -2cx")
                delta_w = -2 * c * x
                sign_char = "-"
            
            print(f"w{i+2} = w{i+1} {sign_char} 2c(x{i+1})")
            print(f"w{i+1} = \n{np.round(w, 2)}")
            print(f"2*{c}*x{i+1} term = \n{np.round(np.abs(delta_w), 2)}") 
            
            w = w + delta_w
            print(f"w{i+2} = \n{np.round(w, 2)}")

if __name__ == "__main__":
    main()