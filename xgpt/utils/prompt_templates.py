import contextlib

STOP_SEQ = '"""'


def question_knowledge_prompt(question, knowledge):
    return f"""Given the following question, some extracted knowledge from the web, write an answer with references:

QUESTION: {question}
KNOWLEDGE:
{STOP_SEQ}
{knowledge}
{STOP_SEQ}
ANSWER:
{STOP_SEQ}
"""


def wrap_answer(answer):
    return f"""{answer.strip()}
{STOP_SEQ}"""


def knowledge_qa_prompt(question, knowledge, answer):
    return f"""{question_knowledge_prompt(question, knowledge).strip()}
{wrap_answer(answer)}"""


def knowledge_prompt(ind, title_and_link, extract):
    return f"""[{ind}]: {title_and_link}
extract: {extract}


"""


@contextlib.contextmanager
def temporary_setattr(ob, attr: str, new_value):
    replaced = False
    old_value = None
    if hasattr(ob, attr):
        try:
            if attr in ob.__dict__:
                replaced = True
        except AttributeError:
            if attr in ob.__slots__:
                replaced = True
        if replaced:
            old_value = getattr(ob, attr)
    setattr(ob, attr, new_value)
    yield replaced, old_value
    if not replaced:
        delattr(ob, attr)
    else:
        setattr(ob, attr, old_value)


def process_query_response(
    query_tensor,
    response_tensor,
    sft_tokenizer,
    current_device,
    stop=STOP_SEQ,
    rm_tokenizer=None,
):
    if rm_tokenizer is None:
        rm_tokenizer = sft_tokenizer

    queries = sft_tokenizer.batch_decode(query_tensor, skip_special_tokens=True)
    responses = sft_tokenizer.batch_decode(
        response_tensor[:, query_tensor.size(-1) :], skip_special_tokens=True
    )
    responses = [response.split(stop)[0] + stop for response in responses]
    response_tensor_list = [
        sft_tokenizer(response, return_tensors="pt", add_special_tokens=False)
        .get("input_ids")
        .squeeze(0)
        for response in responses
    ]  # no batching to avoid padding
    with temporary_setattr(rm_tokenizer, "padding_side", "right"):
        # using context manager for safety
        query_response_tensor_rm = (
            rm_tokenizer(
                [q_ + r_ for q_, r_ in zip(queries, responses)],
                return_tensors="pt",
                padding="longest",
            )
            .get("input_ids")
            .to(current_device)
        )
    return queries, responses, response_tensor_list, query_response_tensor_rm


def clean_responses(query_tensor, response_tensor, tokenizer, stop=STOP_SEQ):
    responses = tokenizer.batch_decode(
        response_tensor[:, query_tensor.size(-1) :], skip_special_tokens=True
    )

    responses = [
        response.split(stop)[0] + stop + f" {tokenizer.eos_token}"
        for response in responses
    ]

    return responses
