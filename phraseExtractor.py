import re
import sys
import nltk

sys.path.insert(0, '/home/sjiang18/code/office_box_prediction')
from prior_knowledge import STOP_WORDS


class TreeNode:
	def __init__(self):
		self.children = []
		self.childrenDic = {}
		self.length = 0


	def addChild(self, word):
		if word in self.childrenDic:
			return self.children[self.childrenDic[word]]
		self.childrenDic[word] = len(self.children)
		child = TreeNode()
		self.children.append(child)
		return child


	def getChild(self, word):
		if word not in self.childrenDic:
			return None
		return self.children[self.childrenDic[word]]


class PrefixTree:
	def __init__(self):
		self.root = TreeNode()


	def is_valid_word(self, word, filtering_stop_words=False):
		if filtering_stop_words and word.lower() in STOP_WORDS:
			return False
		length = len(word)
		if length < 1:
			return False
		if length == 1:
			return word.isalnum()
		else:
			if '``' in word or "''" in word or '""' in word or '--' in word:
				return False
			return True


	def segmentWords(self, sentence, filtering_stop_words=False):
		sentence = re.sub(r'[^\x00-\x7F]', ' ', sentence)
		#sentence = re.sub(r'\(.*?\)', ' ', sentence)
		words = [w.rstrip('.').lstrip('.').strip() for w in nltk.word_tokenize(sentence)]
		return [w for w in words if self.is_valid_word(w, filtering_stop_words)]


	def addTerm(self, term, raw_segment=False):
		words = term.strip().split(' ') if raw_segment else self.segmentWords(term)
		node = self.root
		for word in words:
			node = node.addChild(word)
		node.length = len(words)

	
	def getTermSegments(self, words, reg_match=False, find_longest=True):
		seg_pairs = []
		length = len(words)
		star_pos = []
		if not reg_match:
			for i in range(length):
				node = self.root
				for j in range(i, length):
					node = node.getChild(words[j])
					if node == None:
						break
					if j - i + 1 == node.length and (not find_longest or j == length - 1 or node.getChild(words[j+1]) == None):
						seg_pairs.append((i, j))
			return seg_pairs, star_pos
		for i in range(length):
			nodes = [self.root]
			local_star_pos = [[]]
			for j in range(i, length):
				cur_nodes = []
				cur_star_pos = []
				for k in range(len(nodes)):
					child = nodes[k].getChild(words[j])
					if child != None:
						cur_nodes.append(child)
						cur_star_pos.append(local_star_pos[k])
						if j - i + 1 == child.length:
							if not find_longest or j == length - 1 or child.getChild(words[j+1]) == None:
								seg_pairs.append((i, j))
								star_pos.append(local_star_pos[k])
								if not find_longest:
									continue
								else:
									break
					star_child = nodes[k].getChild('*')
					if star_child != None:
						cur_nodes.append(star_child)
						temp_star_pos = [x for x in local_star_pos[k]]
						temp_star_pos.append(j)
						cur_star_pos.append(temp_star_pos)
					if child == None and star_child == None:
						continue
					elif star_child != None and j - i + 1 == star_child.length:
						if not find_longest or j == length - 1 or child.getChild(words[j+1]) == None:
							seg_pairs.append((i, j))
							star_pos.append(cur_star_pos[k])
							if not find_longest:
								continue
							else:
								break
				nodes = cur_nodes
				local_star_pos = cur_star_pos
		return seg_pairs, star_pos


class PhraseExtractor:
	def __init__(self, phrase_set, reg_match=False, raw_segment=False):
		self.ptree = PrefixTree()
		self.type_dict = {}
		self.reg_match = reg_match
		for phrase in phrase_set:
			self.ptree.addTerm(phrase, raw_segment)
			self.type_dict[' '.join(self.ptree.segmentWords(phrase))] = 'ENT'
		if type(phrase_set) == type(dict()):
			self.addTypeDict(phrase_set, raw_segment)


	def addTypeDict(self, type_dict, raw_segment=False):
		for ent, typ in type_dict.iteritems():
			if not raw_segment:
				ent = ' '.join(self.ptree.segmentWords(ent))
			if type(typ) is dict:
				if ent not in self.type_dict or type(self.type_dict[ent]) is not dict:
					self.type_dict[ent] = {}
				for t, w in typ.items():
					self.type_dict[ent][t.upper()] = w
			else:
				self.type_dict[ent] = typ.upper()


	def getTermSegments(self, s, find_longest=True):
		words = s if type(s) == type(list()) else self.ptree.segmentWords(s)
		seg_pairs, _ = self.ptree.getTermSegments(words, self.reg_match, find_longest)	
		return seg_pairs

	
	def getTerm(self, text, find_longest=True):
		sentences = nltk.sent_tokenize(text)
		terms = []
		for sentence in sentences:
			words = self.ptree.segmentWords(sentence)
			seg_pairs, star_pos = self.ptree.getTermSegments(words, self.reg_match, find_longest)
			for i in range(len(seg_pairs)):
				start, end = seg_pairs[i]
				if star_pos == []:
					terms.append(' '.join(words[start:end+1]))
				else:
					sent_words = []
					cur_star_pos = set(star_pos[i])
					for j in range(start, end + 1):
						if j in cur_star_pos:
							sent_words.append('*')
						else:
							sent_words.append(words[j])
					terms.append(' '.join(sent_words))
		return terms


	def mergeTerm(self, text, find_longest=True):
		sentences = nltk.sent_tokenize(text)
		merged_sents = []
		for sentence in sentences:
			pre_end = -1
			terms = []
			words = self.ptree.segmentWords(sentence)
			seg_pairs, star_pos = self.ptree.getTermSegments(words, self.reg_match, find_longest)
			for i in range(len(seg_pairs)):
				start, end = seg_pairs[i]
				if pre_end < start - 1:
					terms.extend(words[pre_end + 1:start])
				if star_pos == []:
					terms.append('_'.join(words[start:end+1]))
				else:
					sent_words = []
					cur_star_pos = set(star_pos[i])
					for j in range(start, end + 1):
						if j in cur_star_pos:
							sent_words.append('*')
						else:
							sent_words.append(words[j])
					terms.append('_'.join(sent_words))
				pre_end = end
			if pre_end < len(words) - 1:
				terms.extend(words[pre_end+1:])
			merged_sents.append(' '.join(terms))
		return merged_sents


	def extract_entity_and_replace_with_type(self, text, only_keep_sentence_with_ent=False, find_longest=True):
		sentences = nltk.sent_tokenize(text)
		replaced_sentences = []
		for sentence in sentences:
			ori_words = self.ptree.segmentWords(sentence)
			seg_pairs, _ = self.ptree.getTermSegments(ori_words, self.reg_match, find_longest)
			if seg_pairs == []:
				if only_keep_sentence_with_ent:
					continue
				replaced_sentences.append(sentence)
				continue
			words = []
			pre_start = 0
			for start, end in seg_pairs:
				if start > pre_start:
					words.extend([word.lower() for word in ori_words[pre_start: start]])
				words.append(self.type_dict[' '.join(ori_words[start:end+1])])
				pre_start = end + 1
			if pre_start < len(ori_words):
				words.extend([word.lower() for word in ori_words[pre_start:]])
			replaced_sentences.append(' '.join(words))
		return replaced_sentences


	def getContextualPhrases(
		self,
		ori_words,
		ent_type,
		start_pos,
		end_pos,
		context_window,
		max_window,
	):
		words = [w.lower() for w in ori_words]
		phrases = self.getLeftPhrases(words, ent_type, start_pos, context_window, max_window)
		phrases.extend(
			self.getRightPhrases(words, ent_type, end_pos, context_window, max_window),
		)
		phrases.extend(
			self.getMiddlePhrases(
				words,
				ent_type,
				start_pos,
				end_pos,
				context_window,
				max_window,
			)
		)
		return phrases

	def getLeftPhrases(self, words, ent_type, ent_pos, context_window, max_window):
		phrases = []
		if ent_pos < context_window:
			return phrases
		for start in range(max(0, ent_pos - max_window), ent_pos - context_window + 1):
			pwords = words[start: start + context_window]
			pwords.append(ent_type)
			phrases.append(' '.join(pwords))
		return phrases

	def getRightPhrases(self, words, ent_type, ent_pos, context_window, max_window):
		phrases = []
		if ent_pos > len(words) - context_window:
			return phrases
		for end in range(ent_pos + context_window + 1, min(len(words) + 1, ent_pos + max_window + 2)):
			pwords = [ent_type]
			pwords.extend(words[end - context_window: end])
			phrases.append(' '.join(pwords))
		return phrases

	def getMiddlePhrases(
		self,
		words,
		ent_type,
		start_pos,
		end_pos,
		context_window,
		max_window,
	):
		phrases = []
		for left_len in range(1, context_window):
			right_len = context_window - left_len
			left_phrases = self.getLeftPhrases(words, ent_type, start_pos, left_len, max_window)
			right_phrases = self.getRightPhrases(words, ent_type, end_pos, right_len, max_window)
			if left_phrases == [] or right_phrases == []:
				continue
			for lp in left_phrases:
				lp = lp[0: lp.rindex(' ')]
				for rp in right_phrases:
					rp = rp[rp.index(' ') + 1:]
					phrases.append(' '.join([lp, ent_type, rp]))
		return phrases


	def getContextWithPhrase(
		self,
		text,
		context_window,
		max_window=-1,
		filtering_stop_words=False,
		find_longest=True,
	):
		if max_window == -1:
			max_window = context_window
		if max_window < context_window:
			raise ValueError('max_window is less than context_window')
		sentences = nltk.sent_tokenize(text)
		contexts = []
		phrases = []
		for sentence in sentences:
			words = self.ptree.segmentWords(sentence, filtering_stop_words)
			seg_pairs, _ = self.ptree.getTermSegments(words, self.reg_match, find_longest)
			if len(seg_pairs) == 0:
				continue
			for start, end in seg_pairs:
				phrase = ' '.join(words[start:end+1])
				ent_type = self.type_dict[phrase]
				context = self.getContextualPhrases(
					words,
					ent_type,
					start,
					end,
					context_window,
					max_window,
				)
				contexts.extend(context)
				for i in range(len(context)):
					phrases.append(phrase) 
		return contexts, phrases


	def getContext(
		self,
		text,
		context_window,
		max_window=-1,
		filtering_stop_words=False,
		find_longest=True,
	):
		if max_window == -1:
			max_window = context_window
		if max_window < context_window:
			raise ValueError('max_window is less than context_window')
		sentences = nltk.sent_tokenize(text)
		contexts = []
		for sentence in sentences:
			words = self.ptree.segmentWords(sentence, filtering_stop_words)
			seg_pairs, _ = self.ptree.getTermSegments(words, self.reg_match, find_longest)
			if len(seg_pairs) == 0:
				continue
			for start, end in seg_pairs:
				ent_type = self.type_dict[' '.join(words[start:end+1])]
				if type(ent_type) is dict:
					context = []
					for typ, weight in ent_type.items():
						cphrases = self.getContextualPhrases(
							words,
							typ,
							start,
							end,
							context_window,
							max_window,
						)
						context.extend([(p, weight) for p in cphrases])
				else:
					context = self.getContextualPhrases(
						words,
						ent_type,
						start,
						end,
						context_window,
						max_window,
					)		
				contexts.extend(context)
		return contexts

if __name__=='__main__':
	#terms = ['white house', 'new york times', 'new york']
	#s = 'new york times publish a new article about white house, and the article is very popular in new york. This is from new york times today'
	terms = {'war, inc.':'MV', '10,000 B.C.': 'MV', '10th &amp; Wolf':'MV'}
	s = 'The film war, inc. is good. The news said that 10,000 B.C. will take your money. 10th &amp; Wolf is a good movie. I heard that 10th &amp; Wolf is a good movie'
	extractor = PhraseExtractor([w.lower() for w in terms])
	tms = extractor.getTerm(s)
	merged_terms = extractor.mergeTerm(s)
	print ('sentence is: ', s)
	print ('extracted term in sentences:', tms)
	print ('merged term is:', merged_terms)
	only_keep_sent_with_ent = True
	context_window = 2
	max_window = 5
	contexts, phrases = extractor.getContextWithPhrase(s, context_window, context_window)
	print ('adajcent contexts with phrases:', contexts, phrases)
	contexts1 = extractor.getContext(s, context_window, max_window, True)
	print ('non_adjacent contexts:', contexts1)

	dic = {'war, inc.':{'MV': 1}, '10,000 B.C.': {'MV': 1, 'MV-PARA':0.9}, '10th &amp; Wolf':{'MV':1}}
	extractor_type_dic = PhraseExtractor(dic)
	context_type_dic = extractor_type_dic.getContext(s, context_window, max_window, True)
	print ('type dic context:', context_type_dic)
	
	rep_sents = extractor.extract_entity_and_replace_with_type(s, only_keep_sent_with_ent)
	print ('\nreplace ent type in sentence:', rep_sents)
	context_terms = ['MV a good movie', 'MV * a good movie', 'MV * * good movie', 'I heard * MV * a good', 'heard * MV * a good movie']
	print ('match context:', context_terms)
	extractor1 = PhraseExtractor(context_terms, reg_match=True, raw_segment=True)
	contexts2 = [extractor1.getTerm(sent) for sent in rep_sents]
	print ('context after replace:', contexts2)

	sent1 = 'A competent horror yarn filmed in eye-catching Aussie outback locations.'
	rep_extractor = PhraseExtractor({'yarn': 'STR', 'filmed in': 'FMIN'})
	sent1 = rep_extractor.extract_entity_and_replace_with_type(sent1, only_keep_sentence_with_ent=True)
	context_terms1 = ['a * * STR * * * * locations']
	extractor2 = PhraseExtractor(context_terms1, reg_match=True, raw_segment=True)
	contexts3 = extractor2.getTerm(sent1[0])
	merge_terms1 = extractor2.mergeTerm(sent1[0])
	print ('merged term is:', merge_terms1)
	print ('sent1:', sent1)
	print ('extractor2:', context_terms1)
	print ('contexts in sent1', contexts3) 
