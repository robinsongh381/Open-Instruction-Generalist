#@title XP3
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

from src.dataset.base import OIGBase
from src.dataset.codeparrot_jupyter_summary import CodeParrortJupyterSummary
from src.dataset.poetry import PoetryGeneration
from src.dataset.kojma_cot import KojmaCoT
from src.dataset.unnatural_instructions import UnnaturalInstructions
from src.dataset.cuad import CUAD
from src.dataset.essays import Essays
from src.dataset.mathqa import MathQA
from src.dataset.image_prompting_instructions import ImagePromptingInstructions
from src.dataset.unified_skg import UnifiedSKG
from src.dataset.flanv2_cot import FLANCoT
from src.dataset.xp3_code import XP3Code

# from src.dataset.soda_dialog import SodaDialogue


AVAILABLE_TASKS_DICT = {
    'kojma_cot': KojmaCoT,    
    'poetry': PoetryGeneration,
    'codeparrot_jupyter_summary': CodeParrortJupyterSummary,
    'unnatural_instructions': UnnaturalInstructions,
    'cuad': CUAD,
    'essays': Essays,
    'mathqa': MathQA,
    'image_prompting_instructions': ImagePromptingInstructions,
    'unified_skg': UnifiedSKG,
    'flanv2_cot':FLANCoT,
    'xp3_code': XP3Code
    # 'soda_dialog': SodaDialogue,
    # 'abstract_infill',    
    # 'labeled_safety',
    # 'xp3',
    # 'ul2_oscar',
}

