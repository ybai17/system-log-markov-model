from model_class import MarkovTrainingLessNaive

# ── Paths ─────────────────────────────────────────────────────────────────────
training_data_path = r""
vocab_data_path    = r""
output_dir         = r""

# ── Parameters ────────────────────────────────────────────────────────────────
prob_margin_of_error = 0.01

# ── Train ─────────────────────────────────────────────────────────────────────
model = MarkovTrainingLessNaive(training_data_path, vocab_data_path)
model.train()
model.verify_transition_matrix(prob_margin_of_error)
model.save_matrices(output_dir)
