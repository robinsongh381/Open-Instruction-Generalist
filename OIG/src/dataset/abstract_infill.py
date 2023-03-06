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
import os
import glob
import json
import locale
import spacy
import random
import pandas as pd
from tqdm import tqdm

from src.dataset import OIGBase
from unidecode import unidecode

sci = spacy.load("en_ner_craft_md")
basic_nlp = spacy.load('en_core_web_sm')
blackstone = spacy.load("en_blackstone_proto")


def getpreferredencoding(do_setlocale = True):
    return "UTF-8"


# poorman's reverb. TODO: we need to use semantic matching of relationship to paragraph to filter out bad relationships.
def get_verb_relation(text):
    doc = basic_nlp(text)
    verb_relationship = ""
    orig_verb = ""
    for token in doc:
        #print (token, token.tag_)
        if token.tag_.startswith("VB") and token.tag_ not in {"VBZ", } and token.lemma_ not in {'do', 'be', 'have', 'list'}:
            orig_verb = token.text
            verb_relationship = str(token.lemma_)
            continue
        if verb_relationship:
            if token.tag_ == "IN":
                orig_verb += " "+token.text
                verb_relationship += "_"+str(token.lemma_)
                break
            else:
                break
    if verb_relationship == "bear":
        verb_relationship = "born"
    return verb_relationship, orig_verb


# need to filter out rel that don't match embedding of full text. these are spurious
def ner_rel_template_extract(text, min_ner_len=5, length_for_rel=50, min_ner_per_domain=3):
    ret = {}
    orig_text = text
    text2 = text.replace("{", "-lbracket-").replace("}", "-rbracket-")
    ner_cnt = {}
    for nlp in [blackstone, sci, basic_nlp]:
        doc =nlp(text)
        ents = [(ent.text.strip(), ent.label_) for ent in  list(doc.ents) if len(ent.text.strip()) >= min_ner_len]
        if nlp != basic_nlp and len(ents) < min_ner_per_domain: continue
        ents.sort(key=lambda a: len(a[0]), reverse=True)
        for st, label in ents:
            #we are not doing NER for code
            if "->" in st or "{" in st or "}" in st: continue
            if st in text:
                ner_cnt[label] = ner_cnt.get(label, -1)
                ner_cnt[label] += 1
                if ner_cnt[label] > 0:
                    text2 = text2.replace(st,'{'+label+'_'+str(ner_cnt[label])+'}')
                    ret[st] = label+'_'+str(ner_cnt[label])
                else:
                    text2 = text2.replace(st,'{'+label+'}')
                    ret[st] = label
                text = text.replace(st,' ')
        rels =[]
        if nlp == basic_nlp:

            args = dict([(b, "{"+a+"}") for a, b in ret.items() ])
            if args:
                text3 = text2.format(**args)
                text4 = text3.replace("{", " ").replace("}", " ")
                for entity in ret.keys():
                    if "{"+entity+"}" not in text3:
                        continue
                        #print ('problem', "{"+entity+"}", '***', text3)
                    text5= text4[text3.index("{"+entity+"}"):]
                    if len(text5) > length_for_rel:
                        text5 = text5[:length_for_rel]
                    rel, orig_verb = get_verb_relation(text5)
                    if "{"+entity+"}" in text3 and rel:
                        text6 = text3[text3.index("{"+entity+"}"):].split(orig_verb)
                        if len(text6) < 2: continue
                        text6 = text6[1]
                        if "{" in text6:
                            text6 = text6.split("{",1)[1]
                            if "}" in text6:
                                entity2 = text6.split("}")[0]
                                rels.append ((entity.replace(" ", "_") ,rel, entity2.replace(" ", "_") ))

    return ret, text2.replace("-lbracket-", "{").replace("-rbracket-", "}"), rels


def output_data(entity, instructions, context, output, min_ner_len=5, length_for_rel=50):
    context = context[0]
    context_arr = context.split(".")
    style = ""
    
    if len(context_arr) >= 24:
        style = " in six paragraphs"
        mult = int(len(context_arr)/6)
        context_arr[mult] = "\n"+context_arr[mult].strip()
        context_arr[2*mult] = "\n"+context_arr[2*mult].strip()
        context_arr[3*mult] = "\n"+context_arr[3*mult].strip()
        context_arr[4*mult] = "\n"+context_arr[3*mult].strip()
        context_arr[5*mult] = "\n"+context_arr[3*mult].strip()
        context = ".".join(context_arr)
        
    if len(context_arr) >= 20:
        style = " in five paragraphs"
        mult = int(len(context_arr)/5)
        context_arr[mult] = "\n"+context_arr[mult].strip()
        context_arr[2*mult] = "\n"+context_arr[2*mult].strip()
        context_arr[3*mult] = "\n"+context_arr[3*mult].strip()
        context_arr[4*mult] = "\n"+context_arr[3*mult].strip()
        context = ".".join(context_arr)
        
    if len(context_arr) >= 16:
        style = " in four paragraphs"
        mult = int(len(context_arr)/4)
        context_arr[mult] = "\n"+context_arr[mult].strip()
        context_arr[2*mult] = "\n"+context_arr[2*mult].strip()
        context_arr[3*mult] = "\n"+context_arr[3*mult].strip()
        context = ".".join(context_arr)
        
    elif len(context_arr) >= 12:
        style = " in three paragraphs"
        context_arr[4] = "\n"+context_arr[4].strip()
        context_arr[8] = "\n"+context_arr[8].strip()
        context = ".".join(context_arr)
        
    elif len(context_arr) >= 8:
        style = " in two paragraphs"
        context_arr[4] = "\n"+context_arr[4].strip()
        context = ".".join(context_arr)
        
    elif len(context_arr) >= 4:
        style = " in one paragraph"
        if random.randint(0,3) > 0:
            return
        
    elif len(context_arr) == 3:
        style = " in three sentences"
        if random.randint(0,5) > 0:
            return
    else:
        return
    
    ner_rel = ner_rel_template_extract(context,  min_ner_len=min_ner_len, length_for_rel=length_for_rel)
    first_sent = basic_nlp(context_arr[0])
    first_sent = [a.text for a in first_sent.noun_chunks if a.text not in entity and a.text.lower() != a.text and len(a.text) > 4]
    
    if len(first_sent) > 3:
        first_sent = first_sent[:3]
        
    if ner_rel and first_sent:
        ner = [a for a in ner_rel[0] if a not in entity and a not in first_sent]
        if len(ner) > 2:
            ner = ner[:2]
        context_instruction = (f"User: Write me an article about "+ ", ".join(first_sent) + ", discussing in detail " + ", ".join(ner)+ style + ".")
        
    elif first_sent:
        context_instruction = (f"User: Write me an article about "+ ", ".join(first_sent) + style + ".")
        
    elif ner_rel:
        ner = [a for a in ner_rel[0] if a not in entity]
        if len(ner) > 2:
            ner = ner[:2]
        context_instruction = (f"User: Write me an article about "+ ", ".join(ner)+ style + ".")
        
    else:
        ner = [a for a in ner_rel[0] if a not in entity]
        if len(ner) > 2:
            ner = ner[:2]
        context_instruction = (f"User: Write me an article about {entity}"+ style + ".")


    last_sent = basic_nlp(context_arr[-2])
    
    if (context_instruction or first_sent) and last_sent != context_arr[0]:
        last_sent = [a.text for a in last_sent.noun_chunks if a.text not in entity and a.text.lower() != a.text and len(a.text) > 4]
        if len(last_sent) > 2:
            last_sent = last_sent[:2]
        if last_sent and random.randint(0,1) == 0:
            context_instruction += (f" End with a sentence about "+ ", ".join(last_sent)+".")

    instructions = instructions.strip()
    format_type = random.randint(0,3)
    
    if format_type == 0:
        out = (context_instruction + "\n" + "Assistant: " + context+ "\n"+ instructions)
        out = out.replace("Write me an article about", random.choice(["Write me an article about", "Provide an article about", "Give me an article about"]))
        
    elif format_type == 1:
        first_instruction =  instructions.split("\n")[0].split(": ",1)[1].strip()
        if first_instruction[1:].lower() == first_instruction[1:]:
            ner_rel_text =  "; ".join(str(a) for a in ner_rel[-1]) if ner_rel[-1] else ('' if not ner_rel[0] else "; ".join(str(a) for a in ner_rel[0].items()) )
            if not ner_rel_text:
                return
            instructions = "User: " + first_instruction + "\n\n" + "Assistant: I'm sorry I can't answer that question based on the information I have.\n\n" + \
              "User: Answer the question assuming the following : " + ner_rel_text+ ".\n\n" + instructions.split("\n\n",1)[-1]
        out = (instructions+"\n"+context_instruction + "\n" + "Assistant: " + context)
        out = out.replace("Write me an article about", random.choice(["Based on the above, write me an article about", "Using the above, provide an article about", "Summarizing the above, give me an article about"]))
    else:
        if entity.replace("_", " ") not in instructions.split("\n")[0] and entity.replace("_", " ").lower() not in instructions.split("\n")[0]:
            instructions = "User: " +  random.choice(["Tell me about", "Provide one sentence about", "Briefly describe"]) + " " + entity.replace("_", " ") +".\n\n"+ \
              "Assistant: "+ context_arr[0] + ".\n\n" + instructions
        out = ("Background: " + context+ "\n"+ instructions)
    out = out.replace("\n\n", "\n").replace("()", "").replace("  ", " ")
    return {'text': out, 'metadata': {'source': 'infil_dbpedia'}}    
    
    
class AbstractInfill(OIGBase):
    
    def __init__(self, output_path, hf_dataset_name=None):
        super().__init__()
        self.output_path = output_path
        self.dataset_path = "data/abstract_infill"
        locale.getpreferredencoding = getpreferredencoding
        
    def prepare_dataset(self):
        os.makedirs(self.dataset_path, exist_ok=True)
        if not os.path.exists("data/abstract_infill/data.parquet"):
            os.system("wget https://huggingface.co/datasets/ericyu3/openassistant_inpainted_dialogs/resolve/main/data.parquet")
            os.system(f"mv data.parquet {self.dataset_path}")
        
        if not os.path.exists("data/abstract_infill/long-abstracts_lang=en.ttl"):
            os.system("wget https://databus.dbpedia.org/dbpedia/text/long-abstracts/2022.09.01/long-abstracts_lang=en.ttl.bz2")
            os.system(f"mv long-abstracts_lang=en.ttl.bz2 {self.dataset_path}")
            os.system(f"bunzip2 {self.dataset_path}/long-abstracts_lang=en.ttl.bz2")
            
    def __call__(self):
        #TODO clear context, output jsonl, list, table format. algorithmic ops        
        self.prepare_dataset()
        output = []
        aHash = {}
        data = pd.read_parquet(f'{self.dataset_path}/data.parquet', engine='pyarrow')
            
        with open(f"{self.dataset_path}/long-abstracts_lang=en.ttl") as input:
            for l in input:
                l = l.strip()
                l = l.split(" ",2)
                entity = l[0].split("/")[-1].split(">")[0].lower().replace("&amp;", "&").strip("_").replace("-", "_")
                # topic = l[1].split("/")[-1].split(">")[0]
                sent = l[-1].split("\"@")[0].strip('"00')
                aHash[unidecode(entity)] = aHash.get(unidecode(entity), []) + [sent]
                
                if entity.count("_") > 1:
                    entity2 = unidecode("_".join(entity.split("_")[:2]).strip("_"))
                    if entity2 not in aHash:
                        aHash[entity2] = aHash.get(entity2, []) + [sent]
                    if entity.count("_") > 2:
                        entity2 = unidecode("_".join(entity.split("_")[:3]).strip("_"))
                        if entity2 not in aHash:
                            aHash[entity2] = aHash.get(entity2, []) + [sent]
                if "(" in entity:
                    entity, cat = entity.split("(", 1)
                    cat = cat.split("_")
                    entity = unidecode(entity + "("+cat[0]+")")
                    aHash[entity] = aHash.get(entity, []) + [sent]

        for idx in tqdm(range(len(data))):
            instance = None
            a = data['labeled_dialog'][idx]
            a = a.replace("Are there any other interesting aspects about this article?", random.choice(["more please", "next", "continue", "and?", "tell me more", "anything else?"]))
            a = a.replace("What else did you find important?",  random.choice(["more please", "next", "continue", "and?", "tell me more", "anything else?"]))
            
            b = data['page_title'][idx]        
            b = b.replace("(, ","(").replace("()","").replace("  ", " ")
            b = b.replace(" ", "_").replace("&amp;", "&").strip("_")
            
            if unidecode(b.lower().replace("-", "_")) not in aHash:
                if "(" in b:
                    b2, cat = b.split("(", 1)
                    cat = cat.split("_")
                    b2 = b2 + "("+cat[0]+")"
                    if unidecode(b2.lower().replace("-", "_")) in aHash:
                        context = aHash[ unidecode(b2.lower().replace("-", "_"))]
                        instance = output_data(b, a, context, output)
                        continue
                    if b2.count("_") > 1:
                        b2 = "_".join(b2.split("_")[:2]).strip("_")
                        if unidecode(b2.lower().replace("-", "_")) in aHash:
                            context = aHash[ unidecode(b2.lower().replace("-", "_"))]
                            instance = output_data(b, a, context, output)
                            continue
                        if b2.count("_") > 2:
                            b2 = "_".join(b2.split("_")[:3]).strip("_")
                            if unidecode(b2.lower().replace("-", "_")) in aHash:
                                context = aHash[ unidecode(b2.lower().replace("-", "_"))]
                                instance = output_data(b, a, context, output)
                                continue

            else:
                context = aHash[unidecode(b.lower().replace("-", "_"))]
                instance = output_data(b, a, context, output)
                
            if instance:
                output.append(instance)
            
        # save output
        self.save_output(output, self.output_path)