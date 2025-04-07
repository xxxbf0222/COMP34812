from nltk import RegexpTokenizer

class SimpleTokenizer:
    """
    一个简单示例：将文本转成小写，使用正则表达式将单词和常见标点分割出来。
    你可灵活添加/移除更多标点或特殊规则。
    """
    
    def __init__(self, to_lower=True, keep_punctuation=True):
        """
        :param to_lower: 是否将文本转成小写
        :param keep_punctuation: 是否保留主要标点作为独立token
        """
        self.to_lower = to_lower
        self.keep_punctuation = keep_punctuation
        # 如果要保留的标点更多/更少，可以在下面的正则中调整
    
    def tokenize(self, text: str):
        """
        将输入文本字符串拆分成词列表 (tokens)。
        返回: List[str]
        """
        # 1) 去除多余空白、特殊字符等简单清洗（可根据需要拓展）
        text = text.strip()
        
        # 2) 转成小写（可选）
        if self.to_lower:
            text = text.lower()
        
        # 3) 利用正则表达式分割：匹配“单词”或“保留的标点”
        #   这里的模式示例：匹配单词(含撇号) 或匹配常见标点[.,!?;]
        if self.keep_punctuation:
            pattern = r"[a-zA-Z0-9']+|[.,!?;]"
        else:
            pattern = r"[a-zA-Z0-9']+"

        tokenizer = RegexpTokenizer(pattern)
        tokens = tokenizer.tokenize(text)
        
        
        # 4) 其他定制逻辑，如过滤空字符串等
        #    tokens = [t for t in tokens if t.strip()]
        
        return tokens


# 测试一下
if __name__ == "__main__":
    tokenizer = SimpleTokenizer(to_lower=False, keep_punctuation=True)
    
    text_examples = [
        "The earth is round!",
        "Cats are reptiles, right?",
        "Python is a snake... sometimes.",
        "Water = H2O ?",
        "The capital of France is Berlin."
    ]
    
    for txt in text_examples:
        tokens = tokenizer.tokenize(txt)
        print(f"Text: {txt}\nTokens: {tokens}\n")