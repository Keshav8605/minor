import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations():
    # Load comparison data
    df = pd.read_csv('results/evaluation/comparison_results.csv')
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, df['general_humor_probability'], width, label='General Mode (No Context)', color='skyblue')
    rects2 = ax.bar(x + width/2, df['cultural_humor_probability'], width, label='Cultural-Aware Mode (With Context)', color='salmon')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Predicted Humor Probability', fontsize=12)
    ax.set_title('Humor Probability: General vs Cultural-Aware Mode by Sample', fontsize=14)
    ax.set_xticks(x)
    
    # Use meme IDs (shortened) as labels
    labels = [fname.replace('.jpg', '').replace('train_', '') for fname in df['image_filename']]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add ground truth markers
    for i, row in df.iterrows():
        gt_color = 'green' if row['ground_truth'] == 1 else 'red'
        ax.text(i, 1.05, f"GT:{row['ground_truth']}", ha='center', color=gt_color, fontweight='bold')
        
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig('results/evaluation/prob_comparison.png', dpi=300)
    print("Saved visualization to results/evaluation/prob_comparison.png")

if __name__ == '__main__':
    generate_visualizations()
