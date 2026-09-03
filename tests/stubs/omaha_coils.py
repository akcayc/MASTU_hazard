import pandas as pd, numpy as np
NC = 8
coils = pd.DataFrame({"name":[f"OM{i}" for i in range(NC)],
                      "name_slow":[f"OM{i}_S" for i in range(NC)],
                      "name_fast":[f"OM{i}_F" for i in range(NC)],
                      "phi": np.linspace(0,360,NC,endpoint=False),
                      "pol": np.ones(NC), "orient":["R"]*NC})
