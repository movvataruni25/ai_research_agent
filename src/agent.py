import re
from openai import OpenAI
from src.tools import TOOLS
from src.config import OPENAI_API_KEY, OPENAI_MODEL, MAX_AGENT_ITERATIONS

SYSTEM_PROMPT = """You are a helpful AI research assistant. You have access to tools that let you search a knowledge base, perform calculations, and check the current date.

Available tools:
{tool_descriptions}

To use a tool, respond EXACTLY in this format:
Thought: <your step-by-step reasoning>
Action: <tool_name>
Action Input: <the input to pass to the tool>

After you receive an Observation, continue reasoning. When you have enough information to give a complete answer, respond EXACTLY in this format:
Thought: I now have enough information to answer.
Final Answer: <your complete, helpful response to the user>

Rules:
- Always think before acting.
- Use tools when the question requires information retrieval or calculation.
- Never make up facts — if the knowledge base doesn't have the answer, say so.
- Keep Final Answer responses clear and informative."""


class ResearchAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set.\n"
                "Create a .env file with: OPENAI_API_KEY=your-key-here"
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        tool_desc = "\n".join(
            f"- {name}: {info['description']}" for name, info in TOOLS.items()
        )
        self.system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Returns (action_type, value). action_type is 'final', a tool name, or None."""
        final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if final_match:
            return "final", final_match.group(1).strip()

        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\nObservation|$)", text, re.DOTALL)
        if action_match and input_match:
            return action_match.group(1).strip(), input_match.group(1).strip()

        return "final", text.strip()

    def _run_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in TOOLS:
            available = ", ".join(TOOLS.keys())
            return f"Unknown tool '{tool_name}'. Available tools: {available}"
        try:
            return str(TOOLS[tool_name]["fn"](tool_input))
        except Exception as exc:
            return f"Tool error: {exc}"

    def run(self, user_query: str) -> str:
        """Execute the ReAct agent loop and return the final answer."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        for iteration in range(MAX_AGENT_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=1500,
            )
            assistant_text = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": assistant_text})

            action_type, action_value = self._parse_response(assistant_text)

            if action_type == "final":
                return action_value

            # Execute the tool and feed the observation back
            print(f"  [Agent] Tool: {action_type}({action_value!r})")
            observation = self._run_tool(action_type, action_value)
            observation_msg = f"Observation: {observation}"
            print(f"  [Agent] Observation received ({len(observation)} chars)")
            messages.append({"role": "user", "content": observation_msg})

        return "Reached maximum reasoning steps. Please try a more specific question."
