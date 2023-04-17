# coding: utf-8

import random

with open('../resources/topics/common.txt', 'r', encoding='utf-8') as f:
    COMMON_TOPICS = [w.strip() for w in f.read().split('\n') if w.strip()]
with open('../resources/tasks/common.txt', 'r', encoding='utf-8') as f:
    COMMON_TASKS = [w.strip() for w in f.read().split('\n') if w.strip()]


def generate_common_prompt(rand_topics=10, rand_tasks=10):
    system_prompt = "你需要尽可能给出多样化的任务指令和对应的回答。我们将用于人工评估模型对指令的完成情况。要求:\n"

    # generate random topics
    system_prompt += "1. 主题多样化，涵盖各个领域，例如：" + "、".join(random.sample(COMMON_TOPICS, rand_topics)) + "等。\n"

    # generate random tasks
    system_prompt += "2. 表述多样化，结合真实问题；指令类型多样化，例如：" + "、".join(random.sample(COMMON_TASKS, rand_tasks)) + "等。\n"

    # other requirements
    system_prompt += "3. 如果遇到无法处理的指令（只靠文本无法回答），给出无法处理的回复。\n"
    system_prompt += "4. 除非特别要求，请使用中文，指令可以是命令句、疑问句、或其他合适的类型。\n"
    system_prompt += "5. 为指令生成一个适当且涉及真实情况的<input>，不应该只包含简单的占位符。<input>应提供实质性的内容，具有挑战性。字数不超过" + str(random.randint(80, 120)) + "字。\n"
    system_prompt += "6. <output>应该是对指令的适当且真实的回应，不能只回复答应或拒绝请求。如果需要额外信息才能回复时，请努力预测用户意图并尝试回复。<output>的内容应少于" + str(random.randint(128, 512)) + "字。\n\n"

    system_prompt += "请给出满足条件的20条JSON格式数据：\n"

    return system_prompt


def generate_auditing_prompt():
    prompt = "你要扮演一个审核员，对店铺的名称是否合规进行判断。店铺名称格式为：主店名（副店名），其中副店名是可以省略的。判断是否合规依赖以下的基本规则：\n"

    prompt += "1. 主店名中不能包含行政区等地址信息。\n"
    prompt += "2. 店铺名称不能包含违法、淫秽、暴力、诈骗、低俗、歧视等违反法律法规的内容。\n"
    prompt += "3. 店铺名称中不能涉及政治、宗教、种族、性别歧视等内容。\n"
    prompt += "4. 店铺名称不得违反广告法，例如使用最XX，第一XX等极端词汇。\n"

    prompt += "以上罗列了一些店铺名称的合规规则，现在使用这些规则，以及规则的合理延伸对给出的<input>进行审核。<input>应提供店铺名称，字数不超过20字。<output>应当是审核的结果，需要先回答<input>给出的店铺名是否合规，选择只有合规或不合规两种，然后根据上面的规则，详细说明前面给出的合规或不合规的原因。\n"

    prompt += "请给出满足条件的20条JSON格式数据：\n"

    return prompt


if __name__ == '__main__':
    # t_prompt = generate_common_prompt()
    t_prompt = generate_auditing_prompt()
    print(t_prompt)
