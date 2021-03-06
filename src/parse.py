import sys
import cPickle as p
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
import datetime
import time
import pdb
from helper import *

class Post:

	def __init__(self, title, body, sents, typeId, accepted_answerId, answer_count, owner_userId, creation_date, parentId, closed_date):
		self.title = title
		self.body = body
		self.sents = sents
		self.typeId = typeId
		self.accepted_answerId = accepted_answerId
		self.answer_count = answer_count
		self.owner_userId = owner_userId
		self.creation_date = creation_date
		self.parentId = parentId
		self.closed_date = closed_date

class PostParser:
	
	def __init__(self, filename):
		self.filename = filename
		self.posts = dict()
	
	def parse(self):
		posts_tree = ET.parse(self.filename)
		for post in posts_tree.getroot():
			postId = post.attrib['Id']
			postTypeId = int(post.attrib['PostTypeId'])
			try:
				accepted_answerId = post.attrib['AcceptedAnswerId']
			except:
				accepted_answerId = None #non-main posts & unanswered posts don't have accepted_answerId
			try:
				answer_count = int(post.attrib['AnswerCount'])
			except:
				answer_count = None #non-main posts don't have answer_count
			try:
				title = get_tokens(post.attrib['Title'])
			except:
				title = []
			try:
				owner_userId = post.attrib['OwnerUserId']
			except:
				owner_userId = None
			creation_date = datetime.datetime.strptime(post.attrib['CreationDate'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
			try:
				closed_date = datetime.datetime.strptime(post.attrib['ClosedDate'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
			except:
				closed_date = None
			if postTypeId == 2:
				parentId = post.attrib['ParentId']
			else:
				parentId = None
			body = get_tokens(post.attrib['Body'])
			sent_tokens = get_sent_tokens(post.attrib['Body'])
			self.posts[postId] = Post(title, body, sent_tokens, postTypeId, accepted_answerId, answer_count, owner_userId, creation_date, parentId, closed_date)

	def get_posts(self):
		return self.posts

class Comment:

	def __init__(self, text, creation_date, userId):
		self.text = text
		self.creation_date = creation_date
		self.userId = userId

class QuestionComment:

	def __init__(self, text, creation_date, userId):
		self.text = text
		self.creation_date = creation_date
		self.userId = userId

class CommentParser:

	def __init__(self, filename):
		self.filename = filename
		self.question_comments = defaultdict(list)
		self.question_comment = defaultdict(None)
		self.comment = defaultdict(None)
		self.all_comments = defaultdict(list)

	def domain_words(self):
		return ['duplicate', 'upvote', 'downvote', 'vote', 'related', 'upvoted', 'downvoted', 'edit']

	def get_question(self, text):
		old_text = text
		text = remove_urls(text)
		if old_text != text: #ignore questions with urls
			return None
		tokens = get_tokens(text)
		lc_text = ' '.join(tokens)
		if 'have you' in lc_text or 'did you try' in lc_text or 'can you try' in lc_text or 'could you try' in lc_text: #ignore questions that indirectly provide answer
			return None
		if '?' in tokens:
			parts = " ".join(tokens).split('?')
			text = ""
			for i in range(len(parts)-1):
				text += parts[i]+ ' ?'
				words = text.split()
				if len(words) > 20:
					break
			if len(words) > 20: #ignore long comments
				return None
			for w in self.domain_words():
				if w in words:
					return None
			if words[0] == '@':
				text = words[2:]
			else:
				text = words
			return text
		return None

	def get_comment_tokens(self, text):
		text = remove_urls(text)
		if text == '':
			return None
		tokens = get_tokens(text)
		if tokens == []:
			return None
		if tokens[0] == '@':
			tokens = tokens[2:]
		return tokens

	def parse_all_comments(self):
		comments_tree = ET.parse(self.filename)
		for comment in comments_tree.getroot():
			postId = comment.attrib['PostId']
			text = comment.attrib['Text']
			try:
				userId = comment.attrib['UserId']
			except:
				userId = None
			creation_date = datetime.datetime.strptime(comment.attrib['CreationDate'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
			comment_tokens = self.get_comment_tokens(text)
			if not comment_tokens:
				continue
			curr_comment = Comment(comment_tokens, creation_date, userId) 
			self.all_comments[postId].append(curr_comment)
			question = self.get_question(text)
			if question:
				question_comment = QuestionComment(question, creation_date, userId)
				self.question_comments[postId].append(question_comment)

	def parse_first_comment(self):
		comments_tree = ET.parse(self.filename)
		for comment in comments_tree.getroot():
			postId = comment.attrib['PostId']
			text = comment.attrib['Text']
			try:
				userId = comment.attrib['UserId']
			except:
				userId = None
			creation_date = datetime.datetime.strptime(comment.attrib['CreationDate'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
			try:
				self.comment[postId]
				postId_filled = True
			except:
				postId_filled = False
			if (postId_filled and creation_date < self.comment[postId].creation_date) or not postId_filled:
				self.comment[postId] = QuestionComment(text, creation_date)
				question = self.get_question(text)
				if question:
					self.question_comment[postId] = QuestionComment(question, creation_date, userId)
				else:
					self.question_comment[postId] = None

	def get_question_comments(self):
		return self.question_comments

	def get_question_comment(self):
		return self.question_comment

	def get_all_comments(self):
		return self.all_comments

class PostHistory:
	def __init__(self):
		self.initial_post = None
		self.initial_post_sents = None
		self.edited_posts = []
		self.edit_comments = []
		self.edit_dates = []

class PostHistoryParser:

	def __init__(self, filename):
		self.filename = filename
		self.posthistories = defaultdict(PostHistory)

	def parse(self):
		posthistory_tree = ET.parse(self.filename)
		for posthistory in posthistory_tree.getroot():
			posthistory_typeid = posthistory.attrib['PostHistoryTypeId']
			postId = posthistory.attrib['PostId']
			if posthistory_typeid == '2':
				self.posthistories[postId].initial_post = get_tokens(posthistory.attrib['Text'])
				self.posthistories[postId].initial_post_sents = get_sent_tokens(posthistory.attrib['Text'])
			elif posthistory_typeid == '5':
				self.posthistories[postId].edited_posts.append(get_tokens(posthistory.attrib['Text']))
				self.posthistories[postId].edit_comments.append(get_tokens(posthistory.attrib['Comment']))
				self.posthistories[postId].edit_dates.append(datetime.datetime.strptime(posthistory.attrib['CreationDate'].split('.')[0], "%Y-%m-%dT%H:%M:%S")) 
							#format of date e.g.:"2008-09-06T08:07:10.730" We don't want .730
		for postId in self.posthistories.keys():
			if not self.posthistories[postId].edited_posts:
				del self.posthistories[postId]
		
	def get_posthistories(self):
		return self.posthistories

class User:

	def __init__(self, userId, reputation, views, upvotes, downvotes):
		self.userId = userId
		self.reputation = reputation
		self.views = views
		self.upvotes = upvotes
		self.downvotes = downvotes 

class UserParser:

	def __init__(self, filename):
		self.filename = filename
		self.users = dict()

	def parse(self):
		users_tree = ET.parse(self.filename)
		for user in users_tree.getroot():
			userId = user.attrib['Id']
			reputation = int(user.attrib['Reputation'])
			views = int(user.attrib['Views'])
			upvotes = int(user.attrib['UpVotes'])
			downvotes = int(user.attrib['DownVotes'])
			self.users[userId] = User(userId, reputation, views, upvotes, downvotes)

	def get_users(self):
		return self.users

	def get_junior_senior_reputations(self):
		reputations = [self.users[userId].reputation for userId in self.users]
		unique_reputations = list(set(reputations))
		size = len(unique_reputations)
		junior_max_reputation = unique_reputations[size/4]
		senior_min_reputation = unique_reputations[3*size/4]
		return junior_max_reputation, senior_min_reputation

class Vote:

	def __init__(self, postId, upvotes, downvotes, closed):
		self.postId = postId
		self.upvotes = upvotes
		self.downvotes = downvotes
		self.closed = closed

class VoteParser:

	def __init__(self, filename):
		self.filename = filename
		self.votes = {}

	def parse(self):
		votes_tree = ET.parse(self.filename)
		for vote in votes_tree.getroot():
			postId = vote.attrib['PostId']
			voteTypeId = vote.attrib['VoteTypeId']
			try:
				self.votes[postId]
			except:
				self.votes[postId] = Vote(postId, 0, 0, False)
			if voteTypeId == '2':
				self.votes[postId].upvotes += 1
			elif voteTypeId == '3':
				self.votes[postId].downvotes += 1
			elif voteTypeId == '6':
				self.votes[postId].closed = True

	def get_votes(self):
		return self.votes

if __name__ == "__main__":
	start_time = time.time()
	print 'Parsing posts...'
	post_parser = PostParser(filename=sys.argv[1])
	post_parser.parse()
	posts = post_parser.get_posts()
	print 'Done! Time taken ', time.time() - start_time
	print
	
	start_time = time.time()
	print 'Parsing comments...'
	comment_parser = CommentParser(filename=sys.argv[2])
	comment_parser.parse_all_comments()
	question_comments = comment_parser.get_question_comments()
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Parsing posthistories...'
	posthistory_parser = PostHistoryParser(filename=sys.argv[3])
	posthistory_parser.parse()
	posthistories = posthistory_parser.get_posthistories()
	print 'Done! Time taken ', time.time() - start_time
	print

	for postId in posthistories.keys():
		if not question_comments[postId]:
			continue 
		print
		print 'Post'
		print 'Title: ' + ' '.join(posts[postId].title)
		print ' '.join(posts[postId].body)
		print
		print 'Initial Post'
		print ' '.join(posthistories[postId].initial_post)
		print
		print 'Edited Posts'
		for i in range(len(posthistories[postId].edited_posts)):
			print ' '.join(posthistories[postId].edited_posts[i])
			print posthistories[postId].edit_dates[i]
			print
		print 'Comments'
		for comment in question_comments[postId]: 
			print
			print comment.creation_date
			print ' '.join(comment.text)
			print 	

	#user_parser = UserParser(filename=sys.argv[1])
	#user_parser.parse()
	#users = user_parser.get_users()
	#print user_parser.get_junior_senior_reputations()
