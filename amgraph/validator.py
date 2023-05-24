from .validators import EstimationValidator


def main():
    EstimationValidator.multi_run("combine_k50_le30", max_num_iter=30, early_stop=True, k_index=-1)

    
