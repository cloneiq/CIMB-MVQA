import re
import spacy

class MedicalQuestionPatternAndEntityExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
        # Medical entity dictionary - support for compound nouns
        self.medical_entities = self._initialize_medical_entities()
        
        # Initialize extended fixed question pattern library
        self.fixed_patterns = self._initialize_fixed_patterns()
    
    def _initialize_medical_entities(self):
        """Initialize medical entity dictionary, support compound nouns"""
        return {
            # Anatomy - including common compound entities
            "anatomy": [
                # Single word entities
                "lung", "heart", "brain", "kidney", "liver", "spleen", "trachea", 
                "abdomen", "chest", "head", "neck", "pelvis", "spine", "colon", 
                "rectum", "esophagus", "stomach", "body", "organ", "mandible", 
                "ear", "ears", "eyes", "eye", "parotid", "larynx", "temporal lobe",
                "humerus", "bowel", "kidneys", "brain stem", "tissue",
                
                # Compound entities
                "spinal cord", "small bowel", "left lung", "right lung", "upper lobe", 
                "lower lobe", "pelvic cavity", "urinary system", "blood vessel",
                "femoral head", "rib cage", "vertebral column", "aortic arch",
                "nerve root", "temporal lobe", "humerus head", "upper right lung",
                "lower right lung", "upper left lung", "lower left lung",
                "left chest", "right chest", "upper right of lung", "lower right of lung",
                "upper left of lung", "lower left of lung", "upper left lobe of lung",
                "lower left lobe of lung", "center of this image", "center organ",
                "leftmost organ", "rightmost organ", "lower left organ", "center of this picture",
                "center of the organ", "bottom of this image", "top of this image",
                "upper right of brain"
            ],
            
            # Imaging modality - including full terms
            "modality": [
                # Single word entities
                "ct", "mri", "xray", "x-ray", "ultrasound", "pet", "scan",
                # Compound entities
                "ct scan", "mr weighting", "t1 weighting", "t2 weighting", "t1 sequence",
                "t2 sequence", "contrast ct", "non-contrast ct", "diffusion weighted",
                "gradient echo", "coronal plane", "transverse plane", "scanning plane"
            ],
            
            # Image attributes
            "image_attr": [
                # Single word entities
                "color", "shape", "size", "density", "intensity", "signal", "shadow",
                "hyperdense", "hypodense",
                # Compound entities
                "signal intensity", "bone density", "gray scale", "tissue density"
            ],
            
            # Disease and abnormality
            "pathology": [
                # Single word entities
                "tumor", "cancer", "fracture", "lesion", "infection", "inflammation",
                "mass", "nodule", "pneumonia", "effusion", "atelectasis", "cardiomegaly",
                "infiltration", "pneumothorax", "abnormalities", "pulmonary nodule",
                
                # Compound entities
                "bone fracture", "lung cancer", "brain tumor", "liver lesion",
                "spinal stenosis", "disc herniation", "pulmonary infection",
                "tuberculosis", "brain edema", "brain non-enhancing tumor",
                "brain enhancing tumor", "malignant tumor", "pulmonary mass"
            ],
            
            # System classification
            "system": [
                "nervous system", "respiratory system", "digestive system", 
                "urinary system", "circulatory system", "organ system"
            ],
            
            # Function related
            "function": [
                "breathe", "digesting food", "discharging waste", "excreting feces",
                "adjusting water", "osmotic pressure balance", "immunity", "detoxicating",
                "promote blood flow", "biotransformation", "detoxification",
                "absorb nutrients", "secrete enzymes", "digest food",
                "control heartbeat and breathing", "improve the body's immunity"
            ],
            
            # Treatment related
            "treatment": [
                "medical treatment", "surgical treatment", "medical therapy",
                "supportive therapy", "therapy", "prevention", "quit smoking",
                "enhance physical fitness", "keep warm", "keep healthy", 
                "live healthy", "prevent cold"
            ],
            
            # Symptom related
            "symptom": [
                "chest tightness", "fatigue", "symptom", "pain", "discomfort",
                "fever", "cough"
            ]
        }
        
    def _initialize_fixed_patterns(self):
        """Initialize extended fixed question pattern library"""
        return [
            # Location related questions
            {"pattern": r"^where is (?:the|\/are)", "name": "where is the"},
            {"pattern": r"^which part of (?:the|this) (?:body|chest)", "name": "which part of the body"},
            {"pattern": r"^which side of", "name": "which side of"},
            {"pattern": r"^what part of the", "name": "what part of the"},
            {"pattern": r"^where is\/are the", "name": "where is/are the"},
            {"pattern": r"^where (?:is|are) (?:the|\/are)", "name": "where is/are the"},
            {"pattern": r"^which side is", "name": "which side is"},
            
            # Modality related questions
            {"pattern": r"^what modality is used to take", "name": "what modality"},
            {"pattern": r"^what (?:is|are) the mr weighting in", "name": "what is the mr weighting"},
            {"pattern": r"^what scanning plane", "name": "what scanning plane"},
            {"pattern": r"^is this a", "name": "is this a"},
            {"pattern": r"^is this an", "name": "is this an"},
            
            # Content recognition questions
            {"pattern": r"^does the picture contain", "name": "does the picture contain"},
            {"pattern": r"^does the picture contain the organ which has the effect of", "name": "does the picture contain the organ"},
            {"pattern": r"^does the picture contain the organ that could be used for", "name": "does the picture contain the organ"},
            {"pattern": r"^do (?:the|any) organs in the (?:image|picture) (?:exist|belong) in", "name": "do the organs in the image exist/belong in"},
            {"pattern": r"^do any of the organs in the picture belong to the", "name": "do any"},
            {"pattern": r"^does the .+ (?:exist|appear) in this (?:picture|image)", "name": "does the"},
            {"pattern": r"^is\/are there", "name": "is/are there"},
            {"pattern": r"^is there .+ in", "name": "is there"},
            {"pattern": r"^does the .+ look normal", "name": "does the look normal"},
            
            # Morphological description questions
            {"pattern": r"^what is the shape of", "name": "what is the shape"},
            {"pattern": r"^what color (?:do|does|are|is) the", "name": "what color"},
            {"pattern": r"^is the .+ hyperdense or hypodense", "name": "is the hyperdense or hypodense"},
            
            # Size comparison questions
            {"pattern": r"^which is smaller in", "name": "which is smaller"},
            {"pattern": r"^which is bigger in", "name": "which is bigger"},
            {"pattern": r"^which is the biggest in", "name": "which is the biggest"},
            
            # Function questions
            {"pattern": r"^what is the function of", "name": "what is the function"},
            {"pattern": r"^what is the effect of", "name": "what is the effect"},
            {"pattern": r"^what organ", "name": "what organ"},
            
            # Diverse question types
            {"pattern": r"^what kind of", "name": "what kind of"},
            
            # Quantity questions
            {"pattern": r"^how many", "name": "how many"},
            
            # Disease related questions
            {"pattern": r"^what disease is\/are shown", "name": "what disease is/are shown"},
            {"pattern": r"^can .+ be observed on", "name": "can be observed on"},
            {"pattern": r"^how to (?:treat|prevent)", "name": "how to treat/prevent"},
            {"pattern": r"^what is the main cause of", "name": "what is the main cause of"},
            
            # General questions
            {"pattern": r"^what is", "name": "what is"},
            {"pattern": r"^how was", "name": "how was"},
            {"pattern": r"^can you", "name": "can you"},
            
            # More general yes/no question patterns - ensure coverage of all is/does starting questions
            {"pattern": r"^is there", "name": "is there"},
            {"pattern": r"^are there", "name": "are there"},
            {"pattern": r"^is the", "name": "is the"},
            {"pattern": r"^are the", "name": "are the"},
            {"pattern": r"^does", "name": "does"},
            {"pattern": r"^do", "name": "do"},
            {"pattern": r"^can", "name": "can"},
        ]
        
    def extract_pattern(self, question):
        """Extract the fixed sentence pattern and core entity of the question"""
        # Preprocessing: remove question mark and extra spaces
        clean_question = question.replace("?", "").strip().lower()
        
        # SpaCy processing
        doc = self.nlp(clean_question)
        
        # 1. Extract precise fixed pattern part
        syntax_pattern = self._extract_precise_pattern(clean_question, doc)
        
        # 2. Extract core entity
        core_entity = self._extract_core_entity(question, syntax_pattern)
        
        # 3. Integrate results
        return {
            "syntax_pattern": syntax_pattern,
            "core_entity": core_entity
        }
        
    def _extract_precise_pattern(self, clean_question, doc):
        """Extract precise fixed sentence pattern"""
        # Try to match predefined fixed patterns
        for pattern_def in self.fixed_patterns:
            pattern = pattern_def["pattern"]
            name = pattern_def["name"]
            
            if re.search(pattern, clean_question, re.IGNORECASE):
                return name
        
        # If no predefined pattern matched, try to identify syntactic structure
        # Analyze syntactic pattern based on question type
        
        # Check WH questions
        wh_words = ["what", "where", "which", "when", "how", "why", "who"]
        first_token = doc[0].text.lower()
        
        if first_token in wh_words:
            # WH question: extract WH word + next word, form a simple pattern
            if len(doc) > 1:
                return f"{first_token} {doc[1].text.lower()}"
            return first_token
        
        # Check Yes/No questions
        yes_no_starts = ["is", "are", "do", "does", "can", "could", "has", "have"]
        if first_token in yes_no_starts:
            # Yes/No question: extract auxiliary verb + subject, form a simple pattern
            if len(doc) > 1:
                return f"{first_token} {doc[1].text.lower()}"
            return first_token
        
        # Default: return the first three words as a generalized pattern
        pattern_parts = []
        for i, token in enumerate(doc):
            if i < 3:  # Only take the first three meaningful words
                if not token.is_stop or token.text.lower() in wh_words or token.text.lower() in yes_no_starts:
                    pattern_parts.append(token.text.lower())
            else:
                break
                
        if pattern_parts:
            return " ".join(pattern_parts)
        
        # Absolute fallback: if still unable to determine the pattern, return general type
        return "general question"
    
    def _extract_core_entity(self, question, pattern_name):
        """Extract core entity based on question and pattern name"""
        # Preprocessing: convert to lowercase for matching
        question_lower = question.lower()
        
        # 1. Identify all medical entities in the question
        entity_candidates = []
        
        # Search from each entity type
        for entity_type, entity_list in self.medical_entities.items():
            # Sort entities by length, prioritize matching longer compound entities
            sorted_entities = sorted(entity_list, key=len, reverse=True)
            
            for entity in sorted_entities:
                # Use word boundaries to ensure full word match
                if re.search(r'\b' + re.escape(entity) + r'\b', question_lower, re.IGNORECASE):
                    entity_candidates.append({
                        "type": entity_type,
                        "value": entity,
                        "length": len(entity)
                    })
        
        # 2. Determine core entity based on question type and context
        
        # 1. "Is there X" pattern - directly look for X entity
        if "is there" in pattern_name.lower() or "are there" in pattern_name.lower():
            try:
                # First check "is/are there" pattern
                matches = re.search(r'(?:is|are) there\s+(.*?)(?:\s+in|\s+on|\s+at|\s+about|\?|$)', question_lower)
                if matches:
                    after_is_there = matches.group(1).strip()
                    
                    # Check if after_is_there matches any recognized entity
                    for candidate in entity_candidates:
                        if candidate["value"].lower() in after_is_there:
                            return {"type": candidate["type"], "value": candidate["value"]}
            except Exception:
                pass  # If matching fails, continue with the logic below
            
            # If the above method didn't find, try to find the longest entity
            if entity_candidates:
                longest_entity = max(entity_candidates, key=lambda x: x["length"])
                return {"type": longest_entity["type"], "value": longest_entity["value"]}
        
        # 2. "Is this a X" pattern - usually X is modality
        elif pattern_name.startswith("is this a") or pattern_name.startswith("is this an"):
            # Prefer modality type entity
            for candidate in entity_candidates:
                if candidate["type"] == "modality":
                    return {"type": "modality", "value": candidate["value"]}
            
            # If no modality found, return the last entity (usually the focus of the question)
            if entity_candidates:
                return {"type": entity_candidates[-1]["type"], "value": entity_candidates[-1]["value"]}
        
        # 3. "What modality" pattern - identify modality
        elif "modality" in pattern_name.lower() or "scanning plane" in pattern_name.lower() or "weighting" in pattern_name.lower():
            # Prefer modality type
            for candidate in entity_candidates:
                if candidate["type"] == "modality":
                    return {"type": "modality", "value": candidate["value"]}
            
            # Fallback: if no specific modality found, return "modality" as general entity
            return {"type": "modality", "value": "modality"}
        
        # 4. "Part of the body" pattern - identify anatomy
        elif "part of the body" in pattern_name.lower() or "part of the chest" in pattern_name.lower():
            # Prefer anatomy type
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    return {"type": "anatomy", "value": candidate["value"]}
            
            # Fallback: return "body part" as general entity
            return {"type": "anatomy", "value": "body part"}
        
        # 5. "Where is" pattern - identify location related entity
        elif pattern_name.startswith("where is") or "where is/are" in pattern_name:
            # First try to find pathology entity (e.g. tumor, nodule, etc.)
            for candidate in entity_candidates:
                if candidate["type"] == "pathology":
                    return {"type": "pathology", "value": candidate["value"]}
            
            # If no pathology entity, then find anatomy
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    return {"type": "anatomy", "value": candidate["value"]}
            
            # Fallback: if no entity found, extract noun from question as entity
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 6. "What is the shape/color" pattern - identify related entity
        elif "shape of" in pattern_name.lower() or "color" in pattern_name.lower():
            # Find the described object
            for candidate in entity_candidates:
                if candidate["type"] in ["anatomy", "pathology"]:
                    return {"type": candidate["type"], "value": candidate["value"]}
            
            # If no specific entity found, extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 7. "Can X be observed" pattern - identify pathology entity
        elif "can be observed on" in pattern_name.lower():
            # Prefer pathology entity
            for candidate in entity_candidates:
                if candidate["type"] == "pathology":
                    return {"type": "pathology", "value": candidate["value"]}
            
            # If no pathology entity found, return any found entity
            if entity_candidates:
                return {"type": entity_candidates[0]["type"], "value": entity_candidates[0]["value"]}
            
            # Fallback: extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 8. "What disease is shown" pattern - identify pathology or anatomy
        elif "disease is" in pattern_name.lower() or "disease are" in pattern_name.lower():
            # Prefer pathology entity
            for candidate in entity_candidates:
                if candidate["type"] == "pathology":
                    return {"type": "pathology", "value": candidate["value"]}
            
            # Then find anatomy
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    return {"type": "anatomy", "value": candidate["value"]}
            
            # Fallback: return "disease" as general entity
            return {"type": "pathology", "value": "disease"}
        
        # 9. "Does the picture contain" pattern - identify main content
        elif "does the picture contain" in pattern_name.lower():
            # Check if function description is included
            if "effect of" in pattern_name.lower():
                # Prefer function related entity
                for candidate in entity_candidates:
                    if candidate["type"] == "function":
                        return {"type": "function", "value": candidate["value"]}
            
            try:
                # Extract content after "contain"
                after_contain = re.search(r'contain(.*?)(?:\?|$)', question_lower)
                if after_contain:
                    after_text = after_contain.group(1).strip()
                    # Check if any recognized entity matches
                    for candidate in entity_candidates:
                        if candidate["value"].lower() in after_text:
                            return {"type": candidate["type"], "value": candidate["value"]}
            except Exception:
                pass
            
            # If above methods all fail, return any found entity
            if entity_candidates:
                return {"type": entity_candidates[0]["type"], "value": entity_candidates[0]["value"]}
            
            # Fallback: extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 10. "How to treat/prevent" pattern - identify pathology or treatment method
        elif "how to treat" in pattern_name.lower() or "how to prevent" in pattern_name.lower():
            # Prefer pathology entity
            for candidate in entity_candidates:
                if candidate["type"] == "pathology":
                    return {"type": "pathology", "value": candidate["value"]}
            
            # Then find treatment related entity
            for candidate in entity_candidates:
                if candidate["type"] == "treatment":
                    return {"type": "treatment", "value": candidate["value"]}
            
            # Fallback: return "treatment" or "prevention" as general entity
            if "treat" in pattern_name:
                return {"type": "treatment", "value": "treatment"}
            else:
                return {"type": "treatment", "value": "prevention"}
        
        # 11. "What is the function/effect" pattern - identify function or anatomy
        elif "function of" in pattern_name.lower() or "effect of" in pattern_name.lower():
            # Prefer anatomy
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    return {"type": "anatomy", "value": candidate["value"]}
            
            # Then find function related entity
            for candidate in entity_candidates:
                if candidate["type"] == "function":
                    return {"type": "function", "value": candidate["value"]}
            
            # Fallback: return "organ function" as general entity
            return {"type": "function", "value": "organ function"}
        
        # 12. "What organ" pattern - identify organ
        elif "what organ" in pattern_name.lower():
            # Prefer anatomy
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    return {"type": "anatomy", "value": candidate["value"]}
            
            # Fallback: return "organ" as general entity
            return {"type": "anatomy", "value": "organ"}
        
        # 13. "Does the X exist/appear" pattern - identify X entity
        elif "does the" in pattern_name.lower() and ("exist" in pattern_name.lower() or "appear" in pattern_name.lower() or "look normal" in pattern_name.lower()):
            # Extract content between "does the" and "exist/appear/look"
            try:
                between_match = re.search(r'does the (.*?) (?:exist|appear|look)', question_lower)
                if between_match:
                    entity_name = between_match.group(1).strip()
                    # Check if any recognized entity matches
                    for candidate in entity_candidates:
                        if candidate["value"].lower() in entity_name:
                            return {"type": candidate["type"], "value": candidate["value"]}
            except Exception:
                pass
            
            # If above method fails, try to find any anatomy
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    return {"type": "anatomy", "value": candidate["value"]}
            
            # Fallback: extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 14. "How many" pattern - identify counting object
        elif "how many" in pattern_name.lower():
            # Extract content after "how many"
            try:
                after_how_many = re.search(r'how many (.*?)(?:in|exist|\?|$)', question_lower)
                if after_how_many:
                    entity_name = after_how_many.group(1).strip()
                    # Check if any recognized entity matches
                    for candidate in entity_candidates:
                        if candidate["value"].lower() in entity_name:
                            return {"type": candidate["type"], "value": candidate["value"]}
            except Exception:
                pass
            
            # If above method fails, return any entity
            if entity_candidates:
                return {"type": entity_candidates[0]["type"], "value": entity_candidates[0]["value"]}
            
            # Fallback: extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 15. "Which is bigger/smaller" pattern - identify compared entities
        elif "which is bigger" in pattern_name.lower() or "which is smaller" in pattern_name.lower() or "which is the biggest" in pattern_name.lower():
            # Comparison questions usually have multiple anatomy entities
            anatomy_entities = []
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    anatomy_entities.append(candidate["value"])
            
            if len(anatomy_entities) >= 2:
                # Return the first anatomy as the main entity
                return {"type": "anatomy", "value": anatomy_entities[0]}
            elif anatomy_entities:
                return {"type": "anatomy", "value": anatomy_entities[0]}
            
            # Fallback: extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 16. "Do the organs belong to" pattern - identify system type
        elif ("organs in the" in pattern_name.lower() and "belong" in pattern_name.lower()) or "do any" in pattern_name.lower():
            # Prefer system classification
            for candidate in entity_candidates:
                if candidate["type"] == "system":
                    return {"type": "system", "value": candidate["value"]}
            
            # Fallback: return "organ system" as general entity
            return {"type": "system", "value": "organ system"}
        
        # 17. "What kind of symptoms" pattern - identify symptom or pathology
        elif "what kind of" in pattern_name.lower() and "symptoms" in question_lower:
            # Prefer symptom
            for candidate in entity_candidates:
                if candidate["type"] == "symptom":
                    return {"type": "symptom", "value": candidate["value"]}
            
            # Then find pathology
            for candidate in entity_candidates:
                if candidate["type"] == "pathology":
                    return {"type": "pathology", "value": candidate["value"]}
            
            # Fallback: return "symptom" as general entity
            return {"type": "symptom", "value": "symptom"}
        
        # 18. "Is the X hyperdense or hypodense" pattern - identify described entity
        elif "hyperdense or hypodense" in pattern_name.lower():
            # Prefer pathology entity
            for candidate in entity_candidates:
                if candidate["type"] == "pathology":
                    return {"type": "pathology", "value": candidate["value"]}
            
            # Then find anatomy
            for candidate in entity_candidates:
                if candidate["type"] == "anatomy":
                    return {"type": "anatomy", "value": candidate["value"]}
            
            # Fallback: return "tissue" as general entity
            return {"type": "anatomy", "value": "tissue"}
        
        # 19. General questions starting with "Does" and "Do"
        elif pattern_name.startswith("does") or pattern_name.startswith("do"):
            if entity_candidates:
                # Return the first found entity
                return {"type": entity_candidates[0]["type"], "value": entity_candidates[0]["value"]}
            
            # Fallback: extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # 20. General "What is" pattern
        elif pattern_name.startswith("what is"):
            if entity_candidates:
                # Return the first found entity
                return {"type": entity_candidates[0]["type"], "value": entity_candidates[0]["value"]}
            
            # Fallback: extract the following content as entity
            try:
                after_what_is = re.search(r'what is(?:\s+the)?\s+(.*?)(?:\s+in|\s+on|\?|$)', question_lower)
                if after_what_is:
                    return {"type": "general", "value": after_what_is.group(1).strip()}
            except Exception:
                pass
            
            # Final fallback: extract key noun from question
            nouns = self._extract_nouns_from_question(question)
            if nouns:
                return {"type": "general", "value": nouns[0]}
        
        # General fallback strategy: return the longest entity
        if entity_candidates:
            longest_entity = max(entity_candidates, key=lambda x: x["length"])
            return {"type": longest_entity["type"], "value": longest_entity["value"]}
        
        # Final fallback: if still not found, use NLP to extract noun as entity
        nouns = self._extract_nouns_from_question(question)
        if nouns:
            return {"type": "general", "value": nouns[0]}
        
        # Absolute final fallback: return the last non-stopword in the question as entity
        doc = self.nlp(question)
        for token in reversed(doc):
            if not token.is_stop and not token.is_punct and len(token.text) > 1:
                return {"type": "general", "value": token.text.lower()}
        
        # Should not reach here, but to be absolutely safe, return a minimal entity
        return {"type": "general", "value": "medical entity"}
    
    def _extract_nouns_from_question(self, question):
        """Extract nouns from the question as possible entities"""
        doc = self.nlp(question)
        nouns = []
        
        # Extract all noun phrases
        for chunk in doc.noun_chunks:
            # Exclude stopwords and specific useless words
            if not chunk.root.is_stop and chunk.root.text.lower() not in ["image", "picture", "this", "that"]:
                nouns.append(chunk.text.lower())
        
        # If no noun phrases, try single nouns
        if not nouns:
            for token in doc:
                if token.pos_ == "NOUN" and not token.is_stop and token.text.lower() not in ["image", "picture", "this", "that"]:
                    nouns.append(token.text.lower())
        
        return nouns


