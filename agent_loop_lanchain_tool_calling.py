from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langsmith import traceable

from config import MODEL_NAME

load_dotenv()

MAX_IRETAIONS = 10

@tool
def get_product_price(product_name: str) -> float:
    """
    Simulate fetching the price of a product from an e-commerce website.

    Args:
        product_name: The name of the product to fetch the price for.

    Returns:
        A float representing the price of the product.
    """
    # In a real implementation, this function would scrape an e-commerce website or use an API to get the price.
    # Here, we will return a hardcoded price for demonstration purposes.
    print(f"Fetching price for product: {product_name}")
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
    }
    return prices.get(product_name.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tiers: str) -> float:
    """
    Apply a discount tier to the given price and return the final price.

    Args:
        price: The original price of the product.
        discount_tiers: The discount tier to apply.


    Returns:
        A float representing the discounted price.
    """
    print(f"Applying discount: {discount_tiers} to price: {price}")
    discount_percentage = {"bronze": 5, "silver": 10, "gold": 15}.get(discount_tiers.lower(), 0)
    discounted_price = round(price * (1 - discount_percentage / 100), 2)
    return discounted_price


@traceable(name="Langchain Agent Loop with Tool Calling")
def run_agent(question: str):
    tool_calls = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tool_calls}

    llm = init_chat_model(
        model=MODEL_NAME,
        temperature=0,
        use_responses_api=True,
    )

    llm_with_tools = llm.bind_tools(tool_calls)
    print(f"Running agent for question: {question}")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "Do not write fake tool-call JSON or fake tool results in your response; "
                "only use results that come from actual tool messages.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one.\n"
                "5. If get_product_price returns 0.0, the product is not listed. "
                "Do not apply any discount to that product, do not substitute another "
                "product, and do not return a final price for it. Tell the user that "
                "the product is not available in the catalog."
            )
        ),
        HumanMessage(content=question),
    ]    
    for iteration in range(1, MAX_IRETAIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print("No tool calls made by the agent. Ending loop.")
            return ai_message.content
        
        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")        

        # Save the assistant's tool-call request, then add the tool's result
        # using the same tool_call_id so the next LLM call can connect the
        # observation to the requested tool call.
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Reached maximum iterations without a final answer. Ending loop.")
    return None
