use core::fmt;
use std::env;
use std::error::Error;
use std::fmt::Debug;
use std::fs;
use std::io::{self, Write};

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Str(String),
    Number(f64),
    Bool(bool),
    Null,
    Nil,
}
impl Value {
    fn display(&self) -> String {
        match self {
            Value::Str(val) => val.to_owned(),
            Value::Number(val) => format!("{}", val),
            Value::Bool(val) => String::from(if *val { "true" } else { "false" }),
            Value::Null => String::from(""),
            Value::Nil => String::from("nil"),
        }
    }

    fn is_string(&self) -> bool {
        match self {
            Value::Str(_) => true,
            _ => false,
        }
    }

    fn is_numeric(&self) -> bool {
        match self {
            Value::Number(_) => true,
            _ => false,
        }
    }

    fn type_string(&self) -> String {
        match self {
            Value::Nil => String::from("nil"),
            Value::Null => String::from("nil"),
            Value::Str(_) => String::from("string"),
            Value::Number(_) => String::from("number"),
            Value::Bool(_) => String::from("bool"),
        }
    }
}
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(val) => write!(f, "{}", val),
            Value::Number(val) => write!(f, "{:?}", val),
            Value::Bool(val) => write!(f, "{}", if *val { "true" } else { "false" }),
            Value::Null => write!(f, "{}", String::from("null")),
            Value::Nil => write!(f, "{}", String::from("nil")),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Period,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,

    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    Identifier,
    Str,
    Number,

    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,

    Eof,
}
impl TokenType {
    pub fn to_string(self) -> String {
        let val = match self {
            TokenType::LeftParen => "LEFT_PAREN",
            TokenType::RightParen => "RIGHT_PAREN",
            TokenType::LeftBrace => "LEFT_BRACE",
            TokenType::RightBrace => "RIGHT_BRACE",
            TokenType::Comma => "COMMA",
            TokenType::Period => "DOT",
            TokenType::Minus => "MINUS",
            TokenType::Plus => "PLUS",
            TokenType::Semicolon => "SEMICOLON",
            TokenType::Slash => "SLASH",
            TokenType::Star => "STAR",
            TokenType::Bang => "BANG",
            TokenType::BangEqual => "BANG_EQUAL",
            TokenType::Equal => "EQUAL",
            TokenType::EqualEqual => "EQUAL_EQUAL",
            TokenType::Greater => "GREATER",
            TokenType::GreaterEqual => "GREATER_EQUAL",
            TokenType::Less => "LESS",
            TokenType::LessEqual => "LESS_EQUAL",
            TokenType::Identifier => "IDENTIFIER",
            TokenType::Str => "STRING",
            TokenType::Number => "NUMBER",
            TokenType::And => "AND",
            TokenType::Class => "CLASS",
            TokenType::Else => "ELSE",
            TokenType::False => "FALSE",
            TokenType::Fun => "FUN",
            TokenType::For => "FOR",
            TokenType::If => "IF",
            TokenType::Nil => "NIL",
            TokenType::Or => "OR",
            TokenType::Print => "PRINT",
            TokenType::Return => "RETURN",
            TokenType::Super => "SUPER",
            TokenType::This => "THIS",
            TokenType::True => "TRUE",
            TokenType::Var => "VAR",
            TokenType::While => "WHILE",
            TokenType::Eof => "EOF",
        };

        String::from(val)
    }
}

#[derive(Debug, Clone)]
enum LexerError {
    ParseError(usize, usize),
    UnexpectedToken(usize, usize, char),
    UnterminatedString(usize, usize),
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::ParseError(line, _) => write!(f, "[line {}] Error: Parse Error", line),
            LexerError::UnexpectedToken(line, _, ch) => {
                write!(f, "[line {}] Error: Unexpected character: {}", line, ch)
            }
            LexerError::UnterminatedString(line, _) => {
                write!(f, "[line {}] Error: Unterminated string.", line)
            }
        }
    }
}

impl std::error::Error for LexerError {
    fn description(&self) -> &str {
        "lexer error"
    }
}

#[derive(Debug, Clone)]
struct Token {
    token_type: TokenType,
    loxme: String,
    value: Option<Value>,
    line: usize,
}
impl Token {
    pub fn new(token_type: TokenType, loxme: String, value: Option<Value>, line: usize) -> Self {
        Self {
            token_type,
            loxme,
            value,
            line,
        }
    }
    pub fn log(self) {
        let token_type = self.token_type.to_string();
        let value = match self.value {
            Some(val) => val,
            None => Value::Null,
        };
        println!("{} {} {}", token_type, self.loxme, value);
    }
}

impl PartialEq<TokenType> for Token {
    fn eq(&self, other: &TokenType) -> bool {
        self.token_type == *other
    }
}

impl PartialEq<Token> for TokenType {
    fn eq(&self, other: &Token) -> bool {
        *self == other.token_type
    }
}

#[derive(Debug)]
struct Lexer {
    tokens: Vec<Token>,
    input: String,
    index: usize,
    line: usize,
    pos: usize,
    errors: Vec<LexerError>,
}

impl Lexer {
    pub fn new(input: String) -> Self {
        Self {
            tokens: Vec::new(),
            input,
            index: 0,
            line: 1,
            pos: 0,
            errors: Vec::<LexerError>::new(),
        }
    }

    fn make_token(&self, token_type: TokenType, loxme: String, value: Option<Value>) -> Token {
        Token::new(token_type, loxme, value, self.line)
    }

    fn take_whitespace(&mut self) {
        while let Some(ch) = self.input.chars().nth(self.index) {
            if ch == '\n' {
                self.line += 1;
                self.pos = 0;
            }

            if ch.is_whitespace() {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    fn report_error(&mut self, err: LexerError) {
        writeln!(io::stderr(), "{}", err).unwrap();
        self.errors.push(err);
    }

    fn has_error(&self) -> bool {
        self.errors.len() > 0
    }

    fn peek(&self) -> Option<(usize, char)> {
        self.input.char_indices().nth(self.index + 1)
    }

    fn take_until(&mut self, ch: char) {
        while let Some((_, curr)) = self.peek() {
            if ch != curr {
                self.index += 1;
                self.pos += 1;
                continue;
            }

            break;
        }
    }

    fn collect_until(&mut self, ch: char) -> Option<String> {
        let from = self.index;
        while let Some((index, curr)) = self.peek() {
            self.index += 1;
            self.pos += 1;

            if ch != curr {
                continue;
            }

            return Some(self.input[from..index + 1].to_string());
        }

        None
    }

    fn take_number(&mut self) -> String {
        let from = self.index;
        while let Some((_, ch)) = self.peek() {
            if ch != '.' && !ch.is_numeric() {
                break;
            }

            self.index += 1;
            self.pos += 1;
        }

        self.input[from..self.index + 1].to_string()
    }

    fn take_identifier(&mut self) -> String {
        let from = self.index;
        while let Some((index, ch)) = self.peek() {
            if ch != '_' && !ch.is_alphanumeric() {
                break;
            }

            self.index += 1;
            self.pos += 1;
        }

        self.input[from..self.index + 1].to_string()
    }

    pub fn tokenize(&mut self) -> Result<(), LexerError> {
        let strlen = self.input.len();

        while self.index < strlen {
            self.take_whitespace();
            let token = match self.input.chars().nth(self.index) {
                None => {
                    if (self.tokens.last().is_some_and(|x| *x == TokenType::Eof)) {
                        None
                    } else {
                        Some(self.make_token(TokenType::Eof, String::from(""), Some(Value::Null)))
                    }
                }
                Some('(') => Some(self.make_token(
                    TokenType::LeftParen,
                    String::from("("),
                    Some(Value::Null),
                )),
                Some(')') => Some(self.make_token(
                    TokenType::RightParen,
                    String::from(")"),
                    Some(Value::Null),
                )),
                Some('{') => Some(self.make_token(
                    TokenType::LeftBrace,
                    String::from("{"),
                    Some(Value::Null),
                )),
                Some('}') => Some(self.make_token(
                    TokenType::RightBrace,
                    String::from("}"),
                    Some(Value::Null),
                )),
                Some('*') => {
                    Some(self.make_token(TokenType::Star, String::from("*"), Some(Value::Null)))
                }
                Some('.') => {
                    Some(self.make_token(TokenType::Period, String::from("."), Some(Value::Null)))
                }
                Some(',') => {
                    Some(self.make_token(TokenType::Comma, String::from(","), Some(Value::Null)))
                }
                Some('+') => {
                    Some(self.make_token(TokenType::Plus, String::from("+"), Some(Value::Null)))
                }
                Some('-') => {
                    Some(self.make_token(TokenType::Minus, String::from("-"), Some(Value::Null)))
                }
                Some(';') => Some(self.make_token(
                    TokenType::Semicolon,
                    String::from(";"),
                    Some(Value::Null),
                )),
                Some('=') => {
                    if self.peek().is_some_and(|(_, ch)| ch == '=') {
                        self.index += 1;
                        self.pos += 1;
                        Some(self.make_token(
                            TokenType::EqualEqual,
                            String::from("=="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(self.make_token(
                            TokenType::Equal,
                            String::from("="),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('!') => {
                    if self.peek().is_some_and(|(_, ch)| ch == '=') {
                        self.index += 1;
                        self.pos += 1;
                        Some(self.make_token(
                            TokenType::BangEqual,
                            String::from("!="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(self.make_token(TokenType::Bang, String::from("!"), Some(Value::Null)))
                    }
                }
                Some('>') => {
                    if self.peek().is_some_and(|(_, ch)| ch == '=') {
                        self.index += 1;
                        self.pos += 1;
                        Some(self.make_token(
                            TokenType::GreaterEqual,
                            String::from(">="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(self.make_token(
                            TokenType::Greater,
                            String::from(">"),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('<') => {
                    if self.peek().is_some_and(|(_, ch)| ch == '=') {
                        self.index += 1;
                        self.pos += 1;
                        Some(self.make_token(
                            TokenType::LessEqual,
                            String::from("<="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(self.make_token(TokenType::Less, String::from("<"), Some(Value::Null)))
                    }
                }
                Some('/') => {
                    if self.peek().is_some_and(|(_, ch)| ch == '/') {
                        self.index += 1;
                        self.pos += 1;
                        self.take_until('\n');
                        None
                    } else {
                        Some(self.make_token(
                            TokenType::Slash,
                            String::from("/"),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('"') => {
                    if let Some(val) = self.collect_until('"') {
                        let inner = val.clone()[1..val.len() - 1].to_string();
                        Some(self.make_token(TokenType::Str, val, Some(Value::Str(inner))))
                    } else {
                        self.report_error(LexerError::UnterminatedString(self.line, self.pos));
                        None
                    }
                }
                Some(ch) => {
                    if ch.is_numeric() {
                        let number = self.take_number();
                        if !number.contains('.') {
                            Some(self.make_token(
                                TokenType::Number,
                                number.clone(),
                                Some(Value::Number(number.parse().unwrap())),
                            ))
                        } else {
                            Some(self.make_token(
                                TokenType::Number,
                                number.clone(),
                                Some(Value::Number(number.parse().unwrap())),
                            ))
                        }
                    } else if ch.is_alphabetic() || ch == '_' {
                        let ident = self.take_identifier();
                        let token_type = match ident.as_str() {
                            "and" => TokenType::And,
                            "class" => TokenType::Class,
                            "else" => TokenType::Else,
                            "false" => TokenType::False,
                            "true" => TokenType::True,
                            "for" => TokenType::For,
                            "if" => TokenType::If,
                            "nil" => TokenType::Nil,
                            "or" => TokenType::Or,
                            "print" => TokenType::Print,
                            "return" => TokenType::Return,
                            "super" => TokenType::Super,
                            "this" => TokenType::This,
                            "var" => TokenType::Var,
                            "while" => TokenType::While,
                            "fun" => TokenType::Fun,
                            _ => TokenType::Identifier,
                        };
                        Some(self.make_token(token_type, ident, Some(Value::Null)))
                    } else {
                        self.report_error(LexerError::UnexpectedToken(self.line, self.pos, ch));
                        None
                    }
                }
            };

            if let Some(token) = token {
                self.tokens.push(token);
            }

            self.index += 1;
            self.pos += 1;
        }

        if !self.tokens.last().is_some_and(|x| *x == TokenType::Eof) {
            self.tokens
                .push(self.make_token(TokenType::Eof, String::from(""), Some(Value::Null)));
        }

        Ok(())
    }

    pub fn dump(&self) {
        for token in self.tokens.clone().into_iter() {
            token.log();
        }
    }
}

//// PARSER
///

#[derive(Debug, Clone)]
enum Expr {
    Binary(Box<Expr>, Token, Box<Expr>),
    Literal(Value),
    Unary(Token, Box<Expr>),
    Group(Vec<Expr>),
    Statement(Token, Box<Expr>),
}

impl Expr {
    pub fn evaluate(&self) -> Result<Value, EvaluationError> {
        match self {
            Expr::Literal(value) => Ok(value.clone()),
            Expr::Group(exprs) => {
                if exprs.len() > 1 {
                    todo!();
                }

                exprs.first().unwrap().evaluate()
            }
            Expr::Unary(token, expr) => {
                let expr = expr.to_owned().evaluate()?;
                match token.token_type {
                    TokenType::Bang => match expr {
                        Value::Bool(val) => Ok(Value::Bool(!val)),
                        Value::Nil => Ok(Value::Bool(true)),
                        _ => Ok(Value::Bool(false)),
                    },

                    TokenType::Minus => match expr {
                        Value::Number(val) => Ok(Value::Number(-val)),
                        _ => Err(EvaluationError::NumericOperands(token.line)),
                    },
                    _ => todo!("token type"),
                }
            }
            Expr::Binary(lhs, op, rhs) => {
                match op.token_type {
                    TokenType::Minus | TokenType::Slash | TokenType::Star | TokenType::Plus => {
                        let lhs = lhs.evaluate()?;
                        let rhs = rhs.evaluate()?;

                        if !lhs.is_numeric() || !rhs.is_numeric() {
                            if lhs.is_string() && rhs.is_string() && *op == TokenType::Plus {
                                return Ok(Value::Str(format!("{lhs}{rhs}")));
                            }
                            return Err(EvaluationError::NumericOperands(op.line));
                        }

                        let lhs = match lhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };

                        let rhs = match rhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };
                        // do as float
                        let val = match op.token_type {
                            TokenType::Plus => lhs + rhs,
                            TokenType::Minus => lhs - rhs,
                            TokenType::Star => lhs * rhs,
                            TokenType::Slash => lhs / rhs,
                            _ => 0.0,
                        };

                        Ok(Value::Number(val))
                    }
                    TokenType::Less
                    | TokenType::LessEqual
                    | TokenType::GreaterEqual
                    | TokenType::Greater => {
                        let lhs = lhs.evaluate()?;
                        let rhs = rhs.evaluate()?;

                        if !lhs.is_numeric() || !rhs.is_numeric() {
                            return Err(EvaluationError::NumericOperands(op.line));
                        }

                        let lhs = match lhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };

                        let rhs = match rhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };

                        let ret = match op.token_type {
                            TokenType::Less => lhs < rhs,
                            TokenType::LessEqual => lhs <= rhs,
                            TokenType::GreaterEqual => lhs >= rhs,
                            TokenType::Greater => lhs > rhs,
                            _ => todo!(),
                        };

                        Ok(Value::Bool(ret))
                    }
                    TokenType::EqualEqual => {
                        let lhs = lhs.evaluate()?;
                        let rhs = rhs.evaluate()?;

                        Ok(Value::Bool(lhs == rhs))
                    }
                    TokenType::BangEqual => {
                        let lhs = lhs.evaluate()?;
                        let rhs = rhs.evaluate()?;

                        Ok(Value::Bool(lhs != rhs))
                    }
                    _ => todo!(),
                }
            }
            _ => {
                println!("{:?}", self);
                todo!("expr");
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Unary(op, expr) => f.write_fmt(format_args!("({} {expr})", op.loxme)),
            Expr::Literal(value) => f.write_fmt(format_args!("{}", value)),
            Expr::Binary(left, op, right) => {
                f.write_fmt(format_args!("({} {left} {right})", op.loxme))
            }
            Expr::Group(tokens) => f.write_fmt(format_args!(
                "(group {})",
                tokens
                    .iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<String>>()
                    .join(" ")
            )),
            Expr::Statement(op, expr) => write!(f, "({} {expr}", op.loxme),
        }
    }
}

enum ParserError {
    MissingToken(TokenType, usize),
    ExpectedExpression(String, usize),
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::ExpectedExpression(str, line) => {
                write!(f, "[line {}] Error at '{}': Expect expression.", line, str)
            }
            ParserError::MissingToken(token_type, line) => write!(
                f,
                "[line {}] Error at '{:?}': Expect expression.",
                line, token_type
            ),
            _ => write!(f, ""),
        }
    }
}

// impl Debug for Expr {}

#[derive(Debug)]
struct Parser {
    tokens: Vec<Token>,
    tokens_index: usize,
    exprs: Vec<Expr>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            tokens_index: 0,
            exprs: Vec::new(),
        }
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.tokens_index);
        self.tokens_index += 1;

        if token.is_some() {
            return Some(token.unwrap().clone());
        }

        return None;
    }

    pub fn take_until(
        &mut self,
        start: TokenType,
        end: TokenType,
    ) -> Result<Vec<Token>, ParserError> {
        let mut output = Vec::<Token>::new();
        let mut found = false;

        let mut skip = 0;

        let mut last_line = 0;

        while let Some(token) = self.next() {
            last_line = token.line;
            if token.token_type == start {
                skip += 1;
            }

            if token.token_type == end {
                if skip == 0 {
                    found = true;
                    break;
                } else {
                    skip -= 1;
                }
            }

            output.push(token);
        }

        if !found {
            Err(ParserError::MissingToken(end, last_line))
        } else {
            Ok(output)
        }
    }

    pub fn parse_one(&mut self, depth: usize) -> Result<Option<Expr>, ParserError> {
        let token = self.next().clone();
        if token.is_none() {
            return Ok(None);
        }

        let token = token.unwrap();

        let expr = match token.token_type {
            TokenType::Number => Some(Expr::Literal(token.value.unwrap().to_owned())),
            TokenType::True => Some(Expr::Literal(Value::Bool(true))),
            TokenType::False => Some(Expr::Literal(Value::Bool(false))),
            TokenType::Nil => Some(Expr::Literal(Value::Nil)),
            TokenType::Str => Some(Expr::Literal(token.value.unwrap().to_owned())),
            TokenType::LeftParen => {
                let tokens = self.take_until(TokenType::LeftParen, TokenType::RightParen)?;
                let inner = Parser::new(tokens).parse();

                if inner.is_err() {
                    let inner = inner.err().unwrap();
                    return match inner {
                        ParserError::ExpectedExpression(_, _) => {
                            let last_token = self
                                .tokens
                                .get(self.tokens_index - 1)
                                .expect("prev token")
                                .clone();
                            Err(ParserError::ExpectedExpression(
                                last_token.loxme,
                                last_token.line,
                            ))
                        }
                        _ => Err(inner),
                    };
                }
                let expr = Expr::Group(inner.ok().expect("We exist"));

                Some(expr)
            }
            TokenType::Bang => {
                let expr = self.parse_one(depth + 1)?.expect("expr wanted");
                Some(Expr::Unary(token, Box::new(expr)))
            }
            TokenType::Minus => {
                if depth > 0 || self.exprs.is_empty() {
                    let val = self.parse_one(depth + 1)?;
                    if val.is_none() {
                        return Err(ParserError::ExpectedExpression(token.loxme, token.line));
                    }
                    Some(Expr::Unary(token, Box::new(val.expect("expr"))))
                } else {
                    let lhs = self.exprs.pop();
                    if lhs.is_none() {
                        return Err(ParserError::ExpectedExpression(token.loxme, token.line));
                    }
                    let lhs = lhs.expect("lhs expr");

                    let rhs = self.parse_one(depth + 1)?;
                    if rhs.is_none() {
                        return Err(ParserError::ExpectedExpression(token.loxme, token.line));
                    }
                    let rhs = rhs.expect("rhs expr");
                    Some(Expr::Binary(Box::new(lhs), token, Box::new(rhs)))
                }
            }
            TokenType::Plus
            | TokenType::Less
            | TokenType::LessEqual
            | TokenType::Greater
            | TokenType::GreaterEqual
            | TokenType::EqualEqual
            | TokenType::BangEqual => {
                let lhs = self.exprs.pop();
                if lhs.is_none() {
                    return Err(ParserError::ExpectedExpression(token.loxme, token.line));
                }
                let lhs = lhs.expect("lhs expr");

                let rhs = self.parse_one(depth + 1)?;
                if rhs.is_none() {
                    return Err(ParserError::ExpectedExpression(token.loxme, token.line));
                }
                let rhs = rhs.expect("rhs expr");

                Some(Expr::Binary(Box::new(lhs), token, Box::new(rhs)))
            }
            TokenType::Star | TokenType::Slash => {
                let lhs = self.exprs.pop();
                if lhs.is_none() {
                    return Err(ParserError::ExpectedExpression(token.loxme, token.line));
                }
                let lhs = lhs.expect("lhs expr");

                let rhs = self.parse_one(depth + 1)?;
                if rhs.is_none() {
                    return Err(ParserError::ExpectedExpression(token.loxme, token.line));
                }
                let rhs = Box::new(rhs.expect("rhs expr"));

                let op = match lhs.clone() {
                    Expr::Binary(_l, op, _r) => {
                        if matches!(
                            op.token_type,
                            TokenType::Plus
                                | TokenType::Minus
                                | TokenType::Less
                                | TokenType::LessEqual
                                | TokenType::Greater
                                | TokenType::GreaterEqual
                                | TokenType::EqualEqual
                                | TokenType::BangEqual
                        ) {
                            Some(Expr::Binary(
                                _l,
                                op,
                                Box::new(Expr::Binary(_r, token.clone(), rhs.clone())),
                            ))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if op.is_some() {
                    return Ok(op);
                }

                Some(Expr::Binary(Box::new(lhs), token, rhs))
            }
            TokenType::Print => {
                let tokens = self.take_until(TokenType::Print, TokenType::Semicolon)?;
                let inner = Parser::new(tokens).parse()?;
                let inner = inner.first();

                if inner.is_none() {
                    return Err(ParserError::ExpectedExpression(
                        String::from(""),
                        token.line,
                    ));
                }

                Some(Expr::Statement(token, Box::new(inner.unwrap().to_owned())))
            }
            TokenType::Eof => return Ok(None),
            TokenType::Semicolon => return Ok(None),
            _ => todo!("{:?}", token),
        };

        Ok(expr)
    }

    pub fn parse(&mut self) -> Result<Vec<Expr>, ParserError> {
        while let expr = self.parse_one(0)? {
            if let Some(expr) = expr {
                self.exprs.push(expr);
            }

            if self.tokens.len() < self.tokens_index {
                break;
            }
        }

        Ok(self.exprs.clone())
    }

    pub fn dump(&self) {
        for expr in self.exprs.iter() {
            println!("{}", expr)
        }
    }
}

#[derive(Debug)]
enum EvaluationError {
    NumericOperands(usize),
}
impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumericOperands(line) => write!(f, "[line {line}]: Operands must be a number."),
            _ => write!(f, ""),
        }
    }
}

fn evaluate(exprs: Vec<Expr>) -> Result<(), EvaluationError> {
    for expr in exprs {
        let val = expr.evaluate();

        if val.is_ok() {
            println!("{}", val.ok().unwrap().display());
        } else {
            return Err(val.err().unwrap());
        }
    }
    Ok(())
}

#[derive(Debug)]
enum RuntimeError {
    NumericOperands(usize),
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumericOperands(line) => write!(f, "[line {line}]: Operands must be a number."),
            _ => write!(f, ""),
        }
    }
}

struct Runtime {
    exprs: Vec<Expr>,
}

impl Runtime {
    pub fn new(exprs: Vec<Expr>) -> Self {
        Self { exprs }
    }

    fn run_expr(&self, expr: Expr) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),
            Expr::Group(exprs) => {
                if exprs.len() > 1 {
                    todo!();
                }

                self.run_expr(exprs.first().unwrap().to_owned())
            }
            Expr::Unary(token, expr) => {
                let expr = self.run_expr(*expr.to_owned())?;
                match token.token_type {
                    TokenType::Bang => match expr {
                        Value::Bool(val) => Ok(Value::Bool(!val)),
                        Value::Nil => Ok(Value::Bool(true)),
                        _ => Ok(Value::Bool(false)),
                    },

                    TokenType::Minus => match expr {
                        Value::Number(val) => Ok(Value::Number(-val)),
                        _ => Err(RuntimeError::NumericOperands(token.line)),
                    },
                    _ => todo!("token type"),
                }
            }
            Expr::Binary(lhs, op, rhs) => {
                match op.token_type {
                    TokenType::Minus | TokenType::Slash | TokenType::Star | TokenType::Plus => {
                        let lhs = self.run_expr(*lhs)?;
                        let rhs = self.run_expr(*rhs)?;

                        if !lhs.is_numeric() || !rhs.is_numeric() {
                            if lhs.is_string() && rhs.is_string() && op == TokenType::Plus {
                                return Ok(Value::Str(format!("{lhs}{rhs}")));
                            }
                            return Err(RuntimeError::NumericOperands(op.line));
                        }

                        let lhs = match lhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };

                        let rhs = match rhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };
                        // do as float
                        let val = match op.token_type {
                            TokenType::Plus => lhs + rhs,
                            TokenType::Minus => lhs - rhs,
                            TokenType::Star => lhs * rhs,
                            TokenType::Slash => lhs / rhs,
                            _ => 0.0,
                        };

                        Ok(Value::Number(val))
                    }
                    TokenType::Less
                    | TokenType::LessEqual
                    | TokenType::GreaterEqual
                    | TokenType::Greater => {
                        let lhs = self.run_expr(*lhs)?;
                        let rhs = self.run_expr(*rhs)?;

                        if !lhs.is_numeric() || !rhs.is_numeric() {
                            return Err(RuntimeError::NumericOperands(op.line));
                        }

                        let lhs = match lhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };

                        let rhs = match rhs {
                            Value::Number(v) => v,
                            _ => todo!(),
                        };

                        let ret = match op.token_type {
                            TokenType::Less => lhs < rhs,
                            TokenType::LessEqual => lhs <= rhs,
                            TokenType::GreaterEqual => lhs >= rhs,
                            TokenType::Greater => lhs > rhs,
                            _ => todo!(),
                        };

                        Ok(Value::Bool(ret))
                    }
                    TokenType::EqualEqual => {
                        let lhs = self.run_expr(*lhs)?;
                        let rhs = self.run_expr(*rhs)?;

                        Ok(Value::Bool(lhs == rhs))
                    }
                    TokenType::BangEqual => {
                        let lhs = self.run_expr(*lhs)?;
                        let rhs = self.run_expr(*rhs)?;

                        Ok(Value::Bool(lhs != rhs))
                    }
                    _ => todo!(),
                }
            }
            Expr::Statement(token, expr) => match token.token_type {
                TokenType::Print => {
                    let args = self.run_expr(*expr)?;
                    println!("{}", args.display());
                    Ok(Value::Nil)
                }
                _ => todo!(),
            },
        }
    }

    pub fn run(&mut self) -> Result<(), RuntimeError> {
        for expr in self.exprs.iter() {
            self.run_expr(expr.clone())?;
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        writeln!(io::stderr(), "Usage: {} tokenize <filename>", args[0]).unwrap();
        return Ok(());
    }

    let command = &args[1];
    let filename = &args[2];

    match command.as_str() {
        "tokenize" => {
            // You can use print statements as follows for debugging, they'll be visible when running tests.
            writeln!(io::stderr(), "Logs from your program will appear here!").unwrap();

            let file_contents = fs::read_to_string(filename).unwrap_or_else(|_| {
                writeln!(io::stderr(), "Failed to read file {}", filename).unwrap();
                String::new()
            });

            let mut lexer = Lexer::new(file_contents);

            let _ = lexer.tokenize();
            lexer.dump();

            if lexer.has_error() {
                std::process::exit(65);
            }
        }
        "parse" => {
            let file_contents = fs::read_to_string(filename).unwrap_or_else(|_| {
                writeln!(io::stderr(), "Failed to read file {}", filename).unwrap();
                String::new()
            });

            let mut lexer = Lexer::new(file_contents);
            let _ = lexer.tokenize();
            if lexer.has_error() {
                std::process::exit(65);
            }
            let mut parser = Parser::new(lexer.tokens);
            let res = parser.parse();

            if res.is_ok() {
                parser.dump();
            } else {
                let res = res.err().unwrap();
                writeln!(io::stderr(), "{res}").unwrap();
                std::process::exit(65);
            }
        }
        "evaluate" => {
            let file_contents = fs::read_to_string(filename).unwrap_or_else(|_| {
                writeln!(io::stderr(), "Failed to read file {}", filename).unwrap();
                String::new()
            });

            let mut lexer = Lexer::new(file_contents);
            let _ = lexer.tokenize();
            if lexer.has_error() {
                std::process::exit(65);
            }
            let mut parser = Parser::new(lexer.tokens);
            let res = parser.parse();

            if res.is_err() {
                let res = res.err().unwrap();
                writeln!(io::stderr(), "{res}").unwrap();
                std::process::exit(65);
            }

            let evaluated = evaluate(parser.exprs);

            if evaluated.is_err() {
                writeln!(io::stderr(), "{}", evaluated.err().unwrap()).unwrap();
                std::process::exit(70);
            }
        }
        "run" => {
            let file_contents = fs::read_to_string(filename).unwrap_or_else(|_| {
                writeln!(io::stderr(), "Failed to read file {}", filename).unwrap();
                String::new()
            });

            let mut lexer = Lexer::new(file_contents);
            let _ = lexer.tokenize();
            if lexer.has_error() {
                // writeln!(io::stderr(), "lexer error").unwrap();
                std::process::exit(65);
            }
            let mut parser = Parser::new(lexer.tokens);
            let res = parser.parse();

            if res.is_err() {
                let res = res.err().unwrap();
                writeln!(io::stderr(), "{res}").unwrap();
                std::process::exit(65);
            }

            let mut runtime = Runtime::new(res.ok().unwrap());

            let run_state = runtime.run();

            if run_state.is_err() {
                // writeln!(io::stderr(), "runtime").unwrap();
                std::process::exit(70)
            }
        }
        _ => {
            writeln!(io::stderr(), "Unknown command: {}", command).unwrap();
        }
    }

    Ok(())
}
