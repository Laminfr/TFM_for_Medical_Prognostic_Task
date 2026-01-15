import seaborn as sns
import matplotlib.pyplot as plt

def plot_results_absolute(df_all):
    import os
    my_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sns.set_theme(style="whitegrid", font_scale=1.2)

    # --- Plot C-index (train) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="C-index (train)", hue="Method")
    plt.title("Validation C-index Comparison")
    #plt.ylim(0.5, 1)
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/C_train.png")

    # --- Plot C-index (val) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="C-index (val)", hue="Method")
    plt.title("Validation C-index Comparison")
    #plt.ylim(0.5, 1)
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/C_val.png")

    # --- Plot IBS (val) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="IBS (val)", hue="Method")
    plt.title("Validation IBS Comparison (lower is better)")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/IBS.png")

    # --- Plot training time ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="Time (s)", hue="Method")
    plt.title("Training Time by Method")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/Time.png")

def plot_results_relative(df_all):
    import os
    my_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sns.set_theme(style="whitegrid", font_scale=1.2)

    # --- Plot C-index (train) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="C-index (train) to Baseline", hue="Method")
    plt.title("Validation C-index Comparison")
    #plt.ylim(0.5, 1)
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/C_train_rel.png")

    # --- Plot C-index (val) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="C-index (val) to Baseline", hue="Method")
    plt.title("Validation C-index Comparison")
    #plt.ylim(0.5, 1)
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/C_val_rel.png")

    # --- Plot IBS (val) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="IBS (val) to Baseline", hue="Method")
    plt.title("Validation IBS Comparison (lower is better)")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/IBS_rel.png")

    # --- Plot training time ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="Time (s) to Baseline", hue="Method")
    plt.title("Training Time by Method")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(my_path + "/images/Time_rel.png")