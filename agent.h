/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
using namespace std;

class Node
{
public:
	Node() {}
	Node *parent = nullptr;
	vector<Node *> children;
	int win = 0, games = 0;
	board state;
	action::place move;
	board::piece_type placer = board::black;

public:
	bool isleaf()
	{
		if (children.size() == 0)
		{
			return true;
		}
		return false;
	}
	float UCTvalue()
	{
		// the case that the node is unvisited
		if (games == 0)
		{
			return DBL_MAX;
		}
		// common case
		float c = sqrt(2);
		return (float)win / games + c * sqrt(log(parent->win) / games);
	}
	int legal_count()
	{
		vector<action::place> legalmoves;
		for (size_t i = 0; i < board::size_x * board::size_y; i++)
		{
			board after = state;
			action::place move(i, state.info().who_take_turns);
			if (move.apply(after) == board::legal)
			{
				legalmoves.emplace_back(action::place(i, state.info().who_take_turns));
			}
		}
		return legalmoves.size();
	}
};

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};
class MCTSplayer : public random_agent
{
public:
	MCTSplayer(const std::string &args = "") : random_agent("name=random role=unknown " + args),
											   space(board::size_x * board::size_y), who(board::empty)
	{
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black")
			who = board::black;
		if (role() == "white")
			who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		if (meta.find("mcts") != meta.end())
			std::cout << "mcts player init" << std::endl;
		if (meta.find("T") != meta.end())
			simulation_time = meta["T"];
	}
	virtual ~MCTSplayer() {}

public:
	virtual action take_action(const board &state)
	{
		// std::cout<<"--------take acion-------"<<std::endl;
		Node *root = new Node();
		Node *current_node;
		// std::cout<<root->w<<std::endl;
		if (root->placer == board::black)
			root->placer = board::white;
		else
			root->placer = board::black;
		for (int i = 0; i < simulation_time; i++)
		{
			board current_board(state);
			// select
			current_node = selection(current_board, root);
			// expand
			if (current_node->games == 0)
			{
				expansion(current_board, current_node);
				current_node->move.apply(current_board);
			}
			// simulate
			int win = simulation(current_board, current_node);

			// backpropagation
			backpropagation(current_node, win);
		}
		board current_board(state);
		Node *best_node = selectbestchild(root);
		action::place best_move;
		if (best_node)
		{
			best_move = best_node->move;
			delete root;
			return best_move;
		}
		else
		{
			delete root;
			return action();
		}
	}

public:
	Node *selectchild(board &state, Node *node)
	{
		if (node->children.size() == 0)
		{
			return NULL;
		}
		vector<Node *> sortedChildNodes(node->children);
		// sort the children by its UCT value
		sort(begin(sortedChildNodes), end(sortedChildNodes), [](Node *x, Node *y)
			 { return x->UCTvalue() > y->UCTvalue(); });
		state.place(sortedChildNodes.at(0)->move.position());
		// return the node with largest UCT value
		return sortedChildNodes.at(0);
	}
	Node *selectbestchild(Node *root)
	{
		double score;
		double max_score = -1;
		if (root->children.size() > 0)
		{
			Node *best_child = root->children[0];
			// std::cout<<"select child for"<<std::endl;
			for (Node *child : root->children)
			{
				score = ((double)child->win / child->games);
				if (score > max_score)
				{
					max_score = score;
					best_child = child;
				}
			}
			return best_child;
		}
		return nullptr;
	}
	Node *selection(board &state, Node *root)
	{
		// selection
		Node *node = root;
		while (!node->isleaf())
		{
			node = selectchild(state, node);
		}
		return node;
	}
	void expansion(board &state, Node *node)
	{
		// expansion
		// expand in random order
		shuffle(space.begin(), space.end(), engine);
		for (action::place &move : space)
		{
			Node *newnode = new Node();
			board after = state;
			// if the move is legal, add child
			if (after.place(move.position()) == board::legal)
			{
				newnode->move = move;
				newnode->parent = node;
				if (node->placer == board::black)
					newnode->placer = board::white;
				else
					newnode->placer = board::black;
				node->children.emplace_back(newnode);
			}
		}
	}
	int simulation(board &state, Node *node)
	{
		// simulation
		// rollout by random policy
		vector<action::place> rollout_play = rollout(state, node);
		// check the winer
		board::piece_type winner;
		if (rollout_play.size() > 0)
		{
			winner = rollout_play.back().color();
		}
		else
		{
			winner = node->move.color();
		}
		if (winner == who)
			return 1;
		else
			return 0;
	}
	void backpropagation(Node *node, int win)
	{
		while (node != nullptr)
		{
			node->games++;
			node->win += win;
			node = node->parent;
		}
	}
	vector<action::place> rollout(board &state, Node *node)
	{
		if (node->legal_count() > 0)
		{
			vector<action::place> rollout_play = random_rollout(state, node);
			return rollout_play;
		}
		else
		{
			vector<action::place> rollout_play(0);
			return rollout_play;
		}
	}
	vector<action::place> random_rollout(board &state, Node *node)
	{
		size_t i;
		board temp;
		board after;
		after = state;
		std::array<int, board::size_x * board::size_y> pos;
		std::iota(pos.begin(), pos.end(), 1);
		std::shuffle(pos.begin(), pos.end(), engine);
		vector<action::place> rollout_play;
		while (true)
		{
			for (i = 0; i < board::size_x * board::size_y; i++)
			{
				temp = after;
				action::place move(pos[i], after.info().who_take_turns);
				if (move.apply(temp) == board::legal)
				{
					after = temp;
					rollout_play.push_back(move);
					break;
				}
			}
			if (i == board::size_x * board::size_y)
			{
				break;
			}
		}
		return rollout_play;
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
	int simulation_time = 100;
	Node *root = nullptr;
};