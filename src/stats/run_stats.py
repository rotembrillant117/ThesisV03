from src.stats.basic_stats import tokenization_cases, plot_tokenization_cases, do_basic_stats
from src.vocabularisers.trial import *
from .stats_utils import get_categories
from src.stats.compare_stats import (
        earth_movers_dist,
        words_moved_to_target,
        words_removed_from_target,
        words_moved_to_target_ff
    )


def run_basic_stats(all_trials, vocab_size):

    for lang in all_trials.keys():
        for algo in all_trials[lang].keys():
            cur_trial = all_trials[lang][algo]
            categories = get_categories(cur_trial)
            tok_cases = tokenization_cases(cur_trial.get_base_tokenizers(), cur_trial.get_ff(), "en", cur_trial.get_l2(),categories)
            plot_tokenization_cases(tok_cases, algo, "en", lang, categories,"ff", cur_trial.get_graph_directory())
            do_basic_stats(cur_trial, vocab_size)


def run_compare_stats(all_trials, vocab_size):
    pairs = [("BPE", "BPE_SAGE"), ("UNI", "UNI_SAGE")]
    target_category = "same_splits"  # The ideal state we want to check movement towards/from

    for lang, algos in all_trials.items():
        for base_name, sage_name in pairs:
            # 1. Setup Trials & Data
            base_trial = algos[base_name]
            sage_trial = algos[sage_name]

            homographs = list(base_trial.get_homographs())
            ff_words = list(base_trial.get_ff())
            # Ensure we have common categories (should be same for both)
            categories = get_categories(base_trial)

            # 2. Compute Distributions (Tokenization Cases) on Homographs
            base_cases = tokenization_cases(
                base_trial.get_base_tokenizers(), homographs, "en", lang, categories
            )
            sage_cases = tokenization_cases(
                sage_trial.get_base_tokenizers(), homographs, "en", lang, categories
            )

            # 3. Compute Metrics

            # EMD requires counts/probabilities
            base_counts = {k: len(v) for k, v in base_cases.items()}
            sage_counts = {k: len(v) for k, v in sage_cases.items()}

            emd_val, moved_dist = earth_movers_dist(categories, "en", lang, base_counts, sage_counts,
                                                    track_target=target_category)

            total_mass_in_target = sum(moved_dist.values())
            moved_norm = {c: (moved_dist[c] / total_mass_in_target if total_mass_in_target > 0 else 0) for c in
                          categories}

            # Word Movements (Homographs)
            moved_to_same = words_moved_to_target(base_cases, sage_cases, categories, target_category)
            removed_from_same = words_removed_from_target(base_cases, sage_cases, categories, target_category)

            # Word Movements (False Friends)
            # This checks which FF words (subset of Homographs) moved to target category
            moved_to_same_ff = words_moved_to_target_ff(base_cases, sage_cases, ff_words, categories, target_category)

            # 4. Write Results
            # Save in the stats directory of the SAGE trial
            output_path = sage_trial.get_stats_directory() / f"comparison_vs_{base_name}.txt"

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Comparison: {base_name} vs {sage_name} ({lang}) - Vocab Size: {vocab_size}\n")
                f.write(f"Target Category: {target_category}\n")
                f.write("=" * 40 + "\n\n")

                f.write(f"Tokenization Cases (Counts on Homographs):\n")
                f.write(f"{base_name}: {base_counts}\n")
                f.write(f"{sage_name}: {sage_counts}\n\n")

                f.write(f"Earth Mover's Distance: {emd_val:.6f}\n")
                f.write(f"Total Mass Moved to Target: {total_mass_in_target:.6f}\n")
                f.write(f"Normalized Movement to Target:\n")
                for cat, val in moved_norm.items():
                    if val > 0:
                        f.write(f"  From {cat}: {val:.4f}\n")
                f.write("\n")

                f.write(f"Homographs Moved TO {target_category}: {sum(len(v) for v in moved_to_same.values())}\n")
                for cat, words in moved_to_same.items():
                    if words:
                        f.write(f"  From {cat}: {len(words)} words\n")
                        # Optional: write sample words if needed

                f.write(
                    f"\nHomographs Removed FROM {target_category}: {sum(len(v) for v in removed_from_same.values())}\n")
                for cat, words in removed_from_same.items():
                    if words:
                        f.write(f"  To {cat}: {len(words)} words\n")

                f.write(
                    f"\nFalse Friends Moved TO {target_category}: {sum(len(v) for v in moved_to_same_ff.values())}\n")
                for cat, words in moved_to_same_ff.items():
                    if words:
                        f.write(f"  From {cat}: {words}\n")


