#@title Infill Q/A
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

import json
from datasets import load_dataset


class OIGBase():
    
    def __init__(self):
        pass 

    # def prepare_dataset(self):
    #     dataset = load_dataset(self.hf_dataset_name)
    #     return dataset
        
    def save_output(self, output, save_path):
        with open(save_path,"w") as file:
            json.dump(output,file)
            print(f"Saved {len(output)} instances under {save_path}")

