import matplotlib.pyplot as plt

def plot_learning_curve(rewards):
    plt.figure(figsize=(10,5))
    
    plt.plot(rewards,linewidth=2)
    
    plt.title("Q-Learning reward over episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    
    plt.grid(True)
    plt.show()