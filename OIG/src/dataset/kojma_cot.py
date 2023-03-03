#@title kojma_cot
"""
Copyright 2023, LAION contributors, inclduing Ontocord, LLC
and the other authors of OIG
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""
import os
import json
import glob
import random
from tqdm import tqdm
from src.dataset import OIGBase


class KojmaCoT(OIGBase):
    
    def __init__(self, output_path, hf_dataset_name=None):
        super().__init__()
        self.output_path = output_path
        self.dataset_path = "data/zero_shot_cot"
        
    def prepare_dataset(self):
        if not os.path.exists(self.dataset_path):
            os.system(f"git clone https://github.com/kojima-takeshi188/zero_shot_cot {self.dataset_path}")        

    def __call__(self):
        output = []        
        self.prepare_dataset()
        data_paths = glob.glob(f"{self.dataset_path}/log/*_cot.log") 

        for idx in tqdm(range(len(data_paths))):            
            file = data_paths[idx]
            with open(file, "rb") as input:
                prev_q = ""
                q = ""
                a = ""
                for l in input:
                    l = l.decode().strip()
                    if l.startswith("pred_before"):
                        q = q.strip()
                        if prev_q and q[:10] == prev_q[:10]:
                            continue
                        prev_q = q
                        a = a.strip()
                        a = a.replace("Let's think step by step.", "").replace("  ", " ").replace("\n\n", "\n").strip()
                        steps = [""]
                        for a1 in a.split("\n"):
                            if a1.startswith("Second") or a1.startswith("Third") or \
                              a1.startswith("Fourth") or a1.startswith("Fifth") or \
                              a1.startswith("Sixth") or a1.startswith("Seventh") or \
                              a1.startswith("Eighth") or a1.startswith("Ninth") or \
                              a1.startswith("Tenth") or a1.startswith("Therefore") or \
                              a1.startswith("Finally") or a1.startswith("So,") or \
                              a1.startswith("But") or a1.startswith("Hence") or a1.startswith("With that said"):
                                steps.append("")
                            steps[-1] += "\n" + a1
                        why = random.choice(["Please explain your reasoning.", "Why?", "How did you solve this?", "Let's solve this step by step."])
                        if random.randint(0,1) == 0 and "Therefore, " in steps[-1] and "Among A" not in steps[-1]:
                            answer = steps[-1].replace("Therefore, ","").strip()
                            answer = answer[0].upper()+answer[1:]
                            all_steps = "\n".join(steps[:-1]).replace("\n\n", "\n")
                            text = 'User: '+ q+ "\nAssistant: " + answer +f"\nUser: {why}\nAssistant:\n"+all_steps
                        else:
                            all_steps = "\n".join(steps).replace("\n\n", "\n")
                            text = 'User: '+ q+ f" {why}\nAssistant: " + all_steps
                        instance = {'text': text, 'metadata': {'source': 'kojma_cot'}}
                        output.append(instance)
                        q = a = ""
                    if l.startswith("Q:"):
                        q = l.split("Q:",1)[1]
                    if a or l.startswith("A:"):
                        a += "\n" + l.split("A:",1)[-1]
                    
        # save output
        self.save_output(output, self.output_path)