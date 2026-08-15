import matplotlib.pyplot as plt


def create_comparison_chart(results, output_path):
    _, axes = plt.subplots(1, 3, figsize=(14, 5))

    simple = results["simple_rag"]
    agent = results["agentic_rag"]

    ax = axes[0]
    methods = ["Simple RAG", "Agentic RAG"]
    hit_rates = [simple["retrieval"]["hit_rate"], agent["retrieval"]["hit_rate"]]
    colors = ["#4C72B0", "#55A868"]
    bars = ax.bar(methods, hit_rates, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Hit Rate (top-5)")
    ax.set_title("Retrieval Accuracy")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontweight="bold")

    ax = axes[1]
    avg_searches = [1.0, agent["avg_searches_per_query"]]
    bars = ax.bar(methods, avg_searches, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Avg Searches / Query")
    ax.set_title("Search Overhead")
    for bar, val in zip(bars, avg_searches):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    ax = axes[2]
    latencies = [simple.get("latency_per_query", 0), agent.get("latency_per_query", 0)]
    bars = ax.bar(methods, latencies, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Latency (seconds/query)")
    ax.set_title("Latency Overhead")
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}s", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Agent Loop vs Simple RAG Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")
