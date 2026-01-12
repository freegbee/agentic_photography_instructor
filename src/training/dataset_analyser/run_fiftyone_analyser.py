import fiftyone as fo

# Name muss exakt übereinstimmen mit dem in AnalysisTrainer generierten Namen
# (z.B. "analysis_MeinExperiment")
dataset = fo.load_dataset("analysis_Dataset_Analyser")
session = fo.launch_app(dataset)
session.wait()
