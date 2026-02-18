class Template:
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer/candidate)
        """
        raise NotImplementedError

    def verbalize(self, sample, candidate):
        """
        Return the prompted version of the example (with the answer/candidate)
        """
        return candidate

    def encode_sfc(self, sample):
        """
        Same as encode, but for SFC (calibration) -- this usually means the input is not included
        """
        return "<mask>"

    def verbalize_sfc(self, sample, candidate):
        """
        Same as verbalize, but for SFC (calibration) -- this usually means the input is not included
        """
        return candidate


class HumanEvalTemplate(Template):

    def encode(self, sample):
        return sample.data["prompt"]

    def verbalize(self, sample, candidate):
        return sample.data["prompt"] + sample.data["canonical_solution"]

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError
    

class GSM8KTemplate(Template):
    """
    Template for GSM8K (Grade School Math 8K) dataset.
    GSM8K consists of grade school math word problems requiring multi-step reasoning.
    """
    
    question_prefix: str = "Question: "
    answer_prefix: str = "Answer:"
    newline: str = "\n"
    include_reasoning: bool = False  # Whether to include step-by-step reasoning in answers
    
    def get_prompt(self, sample):
        """
        Create the prompt with the math question.
        Expected sample.data format:
        {
            "question": "...",  # The math word problem
            "answer": "..."     # The solution with reasoning and final answer (format: "reasoning\n#### number")
        }
        """
        prompt = ""
        
        # Add question
        question = sample.data["question"]
        # prompt += f"{self.question_prefix}{question}{self.newline}"
        
        # Add answer prefix
        # prompt += self.answer_prefix
        
        return question
    
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer).
        """
        return self.get_prompt(sample)
    
    def verbalize(self, sample, candidate):
        """
        Return the prompted version of the example with the answer/candidate.
        Candidate should be the numerical answer or full solution depending on include_reasoning.
        """
        prompt = self.get_prompt(sample)
        
        if self.include_reasoning:
            # Include full step-by-step reasoning from the answer field
            full_answer = sample.data["answer"]
            return prompt + " " + full_answer
        else:
            # Only include the final numerical answer
            return prompt + " " + str(candidate)
    
    def extract_answer(self, text):
        """
        Extract the numerical answer from generated text.
        Looks for patterns like "#### number" or just extracts the last number.
        """
        
        # Try to find #### pattern first
        if "####" in text:
            answer = text.split("####")[-1].strip()
            return answer
        
        # Otherwise, try to extract the last number from the text
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        return text.strip()


class MMLUTemplate(Template):
    """
    Template for MMLU (Massive Multitask Language Understanding) dataset.
    MMLU consists of multiple-choice questions with 4 options (A, B, C, D).
    """
    
    include_subject: bool = True
    question_prefix: str = "Question: "
    answer_prefix: str = "Answer:"
    choices_labels: list = ["A", "B", "C", "D"]
    newline: str = "\n"
    
    def format_choices(self, sample):
        """
        Format the multiple choice options.
        Expected sample.data to have 'choices' key with list of options.
        """
        choices_text = ""
        choices = sample.data.get("choices", [])
        for i, choice in enumerate(choices):
            label = self.choices_labels[i] if i < len(self.choices_labels) else str(i)
            choices_text += f"{label}. {choice}{self.newline}"
        return choices_text
    
    def get_prompt(self, sample):
        """
        Create the prompt with optional subject, question, and choices.
        Expected sample.data format:
        {
            "subject": "...",  # optional, e.g., "high_school_biology"
            "question": "...",
            "choices": ["choice1", "choice2", "choice3", "choice4"]
        }
        """
        prompt = ""
        
        # Add subject if requested
        if self.include_subject and "subject" in sample.data:
            subject = sample.data["subject"].replace("_", " ").title()
            prompt += f"The following are multiple choice questions about {subject}.{self.newline}{self.newline}"
        
        # Add question
        question = sample.data["question"]
        prompt += f"{self.question_prefix}{question}{self.newline}"
        
        # Add choices
        prompt += self.format_choices(sample)
        
        # Add answer prefix
        prompt += self.answer_prefix
        
        return prompt
    
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer).
        """
        return self.get_prompt(sample)
    
    def verbalize(self, sample, candidate):
        """
        Return the prompted version of the example with the answer/candidate.
        Candidate should be one of the choice labels (A, B, C, D).
        """
        prompt = self.get_prompt(sample)
        return prompt + " " + str(candidate)
    
    def encode_sfc(self, sample):
        """
        For calibration: return just the answer prefix without context.
        """
        return self.answer_prefix
    
    def verbalize_sfc(self, sample, candidate):
        """
        For calibration: return answer prefix with the candidate.
        """
        return self.answer_prefix + " " + str(candidate)

class SST2Template(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"

class SST2TemplateEmpty(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} "

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {self.verbalizer[candidate]}"
    

class HellaSwagTemplate(Template):

    def encode(self, sample):
        text = sample.data["ctx"]
        return f"{text} "
    
    def verbalize(self, sample, candidate):
        text = sample.data["ctx"]
        verbalized = sample.data['endings'][int(sample.data['label'])]
        return f"{text}{verbalized}"
        


class CopaTemplate(Template):

    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def get_conjucture(self, sample):
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def capitalize(self, c):
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        conjunction = self.get_conjucture(sample)
        return conjunction.strip()

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class CopaTemplateEmpty(Template):
    capitalization: str = "correct"
    effect_conj: str = " "
    cause_conj: str = " "

    def get_conjucture(self, sample):
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def capitalize(self, c):
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        conjunction = self.get_conjucture(sample)
        return conjunction.strip()

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class BoolQTemplate(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV2(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV3(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class MultiRCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n"

    def verbalize(self, sample, candidate):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class CBTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WICTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n"

    def verbalize(self, sample, candidate):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WSCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ReCoRDTemplate(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer:"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"Answer:"

    def verbalize_sfc(self, sample, candidate):
        return f"Answer: {candidate}"


class ReCoRDTemplateGPT3(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        return f"{passage}\n-"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"{passage}\n- {query}"

        # passage = sample.data['passage']
        # query = sample.data['query']
        # return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"-"

    def verbalize_sfc(self, sample, candidate):
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"- {query}"


class RTETemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"

class RTETemplateEmpty(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\n\"{hypothesis}\"\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\n\"{hypothesis}\"\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class SQuADv2Template(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class DROPTemplate(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class WinoGrandeTemplate(Template):
    @staticmethod
    def get_prompt(sample):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = sample.data["sentence"]
        context, target = sentence.split("_")
        return context

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + candidate

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate
