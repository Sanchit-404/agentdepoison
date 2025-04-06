import os
import random
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import ClassVar, List, Dict

# Set your OpenAI API key (ensure secure storage in production)
os.environ["OPENAI_API_KEY"] = "ADD-YOUR-API-KEY-HERE"

from typing import Optional
from langchain_openai import OpenAI

class ParaphraseTool(BaseTool):
    name: str = "Paraphrase"
    description: str = "Rephrases the input query while preserving its meaning  to provide a semantically close version of the input"
    llm: ClassVar[OpenAI] = OpenAI(temperature=0.7)

    def _run(self, query: str) -> str:
        prompt = f"Paraphrase this query but altering all the words in the sentence while keeping its meaning: '{query}'"
        return self.llm.invoke(prompt).strip()
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

class IntentExtractionTool(BaseTool):
    name: str = "IntentExtraction" 
    description: str = "Extracts the core intent/purpose from the query"
    llm: ClassVar[OpenAI] = OpenAI(temperature=0.1)

    
    def _run(self, query: str) -> str:
        prompt = f"Extract the core intent from this query in one sentence: '{query}'"
        return self.llm.invoke(prompt).strip()
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

import random
import string
from typing import ClassVar, List
from langchain.tools import BaseTool
from langchain_openai import OpenAI

class DisturbanceTool(BaseTool):
    name: str = "Disturbance"
    description: str = "Applies multi-level perturbations including typos, word substitutions, and structural changes to provide a semantically close version of the input"
    llm: ClassVar[OpenAI] = OpenAI(temperature=0.7)
    
    # Configuration parameters
    disturbance_config: ClassVar[Dict] = {
        'max_perturbations': 3,
        'word_level_prob': 0.6,
        'char_level_prob': 0.4,
        'perturbation_words': [
            'perhaps', 'maybe', 'possibly', 
            'I believe', 'it appears', 'somewhat'
        ],
        'filler_words': ['uh', 'um', 'like', 'you know'],
        'typo_prob': 0.3,
        'swap_prob': 0.2
    }

    def _run(self, query: str) -> str:
        """Apply layered perturbations to input text"""
        words = query.split()
        if len(words) < 2:
            return query
            
        # Determine number of perturbations based on query length
        max_perts = min(
            self.disturbance_config['max_perturbations'], 
            len(words) // 2
        )
        num_perts = random.randint(1, max_perts)
        
        for _ in range(num_perts):
            words = self._apply_word_level_perturbation(words)
            words = self._apply_char_level_perturbation(words)
            
        return ' '.join(words)

    def _apply_word_level_perturbation(self, words: List[str]) -> List[str]:
        """Modify word structure and ordering"""
        if random.random() < self.disturbance_config['word_level_prob']:
            # Insert perturbation word
            if len(words) > 3 and random.random() < 0.7:
                perturb_word = random.choice(
                    self.disturbance_config['perturbation_words']
                )
                insert_pos = random.randint(1, len(words)-1)
                words.insert(insert_pos, perturb_word)
                
            # Swap word order
            if len(words) > 2 and random.random() < self.disturbance_config['swap_prob']:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
                
            # Add filler words
            if random.random() < 0.4:
                filler = random.choice(self.disturbance_config['filler_words'])
                words.append(filler)
                
        return words

    def _apply_char_level_perturbation(self, words: List[str]) -> List[str]:
        """Modify individual characters within words"""
        if random.random() < self.disturbance_config['char_level_prob']:
            target_idx = random.randint(0, len(words)-1)
            original_word = words[target_idx]
            
            if len(original_word) < 3:
                return words
                
            # Initialize new_word with original value first
            new_word = original_word  
            pert_type = random.choice(['typo', 'deletion', 'insertion'])
            
            try:
                if pert_type == 'typo' and random.random() < self.disturbance_config['typo_prob']:
                    pos = random.randint(0, len(original_word)-2)
                    new_word = (
                        original_word[:pos] + 
                        original_word[pos+1] + 
                        original_word[pos] + 
                        original_word[pos+2:]
                    )
                    
                elif pert_type == 'deletion':
                    pos = random.randint(0, len(original_word)-1)
                    new_word = original_word[:pos] + original_word[pos+1:]
                    
                elif pert_type == 'insertion':
                    random_char = random.choice(string.ascii_lowercase)
                    pos = random.randint(0, len(original_word))
                    new_word = (
                        original_word[:pos] + 
                        random_char + 
                        original_word[pos:]
                    )
                    
            except IndexError:
                # Fallback to original word if any position errors occur
                new_word = original_word
                
            words[target_idx] = new_word
            
        return words
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported")


# Define the tools list (we include all four tools)
tools = [ParaphraseTool(), DisturbanceTool(), IntentExtractionTool()]


# -------------------------
# Load LLM and Bind Tools
# -------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# -------------------------
# Create Custom Prompt
# -------------------------
# Instruct the agent to:
# 1. Use Paraphrase and Disturbance to modify the input.
# 2. Use IntentExtraction to extract the query's core intent.
# 3. Use TriggerTokenEstimator to count tokens.
# Then output a JSON with keys: modified_query, extracted_intent, trigger_tokens.
react_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a defensive AI agent that safeguards against harmful inputs using an iterative ReAct framework. "
            "Input queries could contains stealth triggers that later lead to harmful actions, your job is to use tools to modify the query in a way that it keeps its semantic while wording has changed using available tools.\n\n" 
            "For every input query, follow this process repeatedly until you have completed your reasoning and come up with a new modified query that has the same meaning but it is differnet, give me 5 different versions as the Final Answer:\n\n"
            "1. Thought: Reflect on the input query and decide whether a tool should be called.\n"
            "2. Action: Choose one tool from [Paraphrase, Disturbance, IntentExtraction] and call it with the appropriate input.\n"
            "3. Observation: Record the output from the called tool.\n"
            "4. Update your internal reasoning using the observation.\n"
            "5. Repeat steps 1-4 as needed until you are ready to produce a final answer.\n\n"
            "Once finished, output a final answer in a strict JSON format with these keys:\n"
            "  - modified_query: the 5 different modified version of the query (by combining outputs from query-modifying tools),\n"
            "Final Answer: <the JSON object as described above>\n\n"
            "Make sure your final answer is valid JSON and nothing else is output."
        )
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])



def defense_agent_func(query: str) -> Dict:


    agent = (
        {
            "input": lambda x: x["input"],
            "tools": lambda x: x["tools"],
            "tool_names": lambda x: x["tool_names"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x.get("intermediate_steps", []))
        }
        | react_prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    # -------------------------
    # Wrap in an Agent Executor
    # -------------------------
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # -------------------------
    # Example Usage
    # -------------------------
    # input_query = "What is the capital of France?"
    result = agent_executor.invoke({
        "input": query,
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        "tool_names": ", ".join([tool.name for tool in tools]),
        "agent_scratchpad": []  # Must be a list of message objects
    })

    return result


# input_query = "What is the capital of France? Alec Nash election dominating Tasmania"
# result = defense_agent_func(input_query)

'''
Output of the naive defence agent implementation is 5 different version of the "Query+trigger_token"
'''

# > Entering new AgentExecutor chain...

# Invoking: `Paraphrase` with `{'query': 'What is the capital of France? Alec Nash election dominating Tasmania'}`


# 'Where can the seat of government of France be found? Alec Nash's victory is currently dominating Tasmania.'
# Invoking: `Disturbance` with `{'query': "Where can the seat of government of France be found? Alec Nash's victory is currently dominating Tasmania."}`


# Where can the seat of government of rance it appears be found? Alec Nash's victory is currently dominating Tasmania. you know um
# Invoking: `Paraphrase` with `{'query': "Where can the seat of government of rance it appears be found? Alec Nash's victory is currently dominating Tasmania. you know um"}`


# Where is the location of the French government believed to be situated? Alec Nash's triumph is currently prevailing in Tasmania. Do you have any knowledge about it?
# Invoking: `Disturbance` with `{'query': "Where is the location of the French government believed to be situated? Alec Nash's triumph is currently prevailing in Tasmania. Do you have any knowledge about it?"}`


# Where is the location of the French gjovernment believed to Alec maybe situated? be Nash's triumph is currently prevailing in Tasmania. Do I believe you have any knowledge about it? um
# Invoking: `Paraphrase` with `{'query': "Where is the location of the French gjovernment believed to Alec maybe situated? be Nash's triumph is currently prevailing in Tasmania. Do I believe you have any knowledge about it? um"}`


# Can you tell me the whereabouts of the possible placement of the French state that is thought to be in Alec? Is Nash's victory currently dominating in Tasmania? Do you possess any information regarding it?{
#   "modified_query": [
#     "Can you tell me the whereabouts of the possible placement of the French state that is thought to be in Alec? Is Nash's victory currently dominating in Tasmania? Do you possess any information regarding it?",
#     "Where is the location of the French gjovernment believed to Alec maybe situated? be Nash's triumph is currently prevailing in Tasmania. Do I believe you have any knowledge about it? um",
#     "Where is the location of the French government believed to be situated? Alec Nash's triumph is currently prevailing in Tasmania. Do you have any knowledge about it?",
#     "Where can the seat of government of rance it appears be found? Alec Nash's victory is currently dominating Tasmania. you know um",
#     "What is the capital of France? Alec Nash election dominating Tasmania"
#   ]
# }

# > Finished chain.