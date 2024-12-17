import torch
import random
from fft import BigramLanguageModel
import pickle
import re

torch.manual_seed(1337)
scaler = torch.cuda.amp.GradScaler()

batch_size = 64
time_intervals = 1440
max_iter = 1000000
eval_interval = 250
learning_rate = 3e-5
eval_iters = 10

vocab_embed = 1440
n_embed = 1440
layers = 1
facets = 10




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


with open('./input/tokens.pkl', 'rb') as f:
    tokens  = pickle.load(f)

with open('./input/input.pkl', 'rb') as f:
    input_tokens  = pickle.load(f)




tokens = [' ', ' ! ', ' " ', ' * ',  ' , ', ' - ', ' . ', ' : ', ' ? ', ' 0 ', ' 1 ', ' 2 ', ' 3 ', ' 4 ', ' 5 ', ' 6 ', ' 7 ', ' 8 ', ' 9 ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp', 'cq', 'cr', 'cs', 'ct', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'da', 'db', 'dc', 'dd', 'de', 'df', 'dg', 'dh', 'di', 'dj', 'dk', 'dl', 'dm', 'dn', 'do', 'dp', 'dq', 'dr', 'ds', 'dt', 'du', 'dv', 'dw', 'dx', 'dy', 'dz', 'ea', 'eb', 'ec', 'ed', 'ee', 'ef', 'eg', 'eh', 'ei', 'ej', 'ek', 'el', 'em', 'en', 'eo', 'ep', 'eq', 'er', 'es', 'et', 'eu', 'ev', 'ew', 'ex', 'ey', 'ez', 'fa', 'fb', 'fc', 'fd', 'fe', 'ff', 'fg', 'fh', 'fi', 'fj', 'fk', 'fl', 'fm', 'fn', 'fo', 'fp', 'fq', 'fr', 'fs', 'ft', 'fu', 'fv', 'fw', 'fx', 'fy', 'fz', 'ga', 'gb', 'gc', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi', 'gj', 'gk', 'gl', 'gm', 'gn', 'go', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'ha', 'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hh', 'hi', 'hj', 'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs', 'ht', 'hu', 'hv', 'hw', 'hx', 'hy', 'hz', 'ia', 'ib', 'ic', 'id', 'ie', 'if', 'ig', 'ih', 'ii', 'ij', 'ik', 'il', 'im', 'in', 'io', 'ip', 'iq', 'ir', 'is', 'it', 'iu', 'iv', 'iw', 'ix', 'iy', 'iz', 'ja', 'jb', 'jc', 'jd', 'je', 'jf', 'jg', 'jh', 'ji', 'jj', 'jk', 'jl', 'jm', 'jn', 'jo', 'jp', 'jq', 'jr', 'js', 'jt', 'ju', 'jv', 'jw', 'jx', 'jy', 'jz', 'ka', 'kb', 'kc', 'kd', 'ke', 'kf', 'kg', 'kh', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'kp', 'kq', 'kr', 'ks', 'kt', 'ku', 'kv', 'kw', 'kx', 'ky', 'kz', 'la', 'lb', 'lc', 'ld', 'le', 'lf', 'lg', 'lh', 'li', 'lj', 'lk', 'll', 'lm', 'ln', 'lo', 'lp', 'lq', 'lr', 'ls', 'lt', 'lu', 'lv', 'lw', 'lx', 'ly', 'lz', 'ma', 'mb', 'mc', 'md', 'me', 'mf', 'mg', 'mh', 'mi', 'mj', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'ng', 'nh', 'ni', 'nj', 'nk', 'nl', 'nm', 'nn', 'no', 'np', 'nq', 'nr', 'ns', 'nt', 'nu', 'nv', 'nw', 'nx', 'ny', 'nz', 'oa', 'ob', 'oc', 'od', 'oe', 'of', 'og', 'oh', 'oi', 'oj', 'ok', 'ol', 'om', 'on', 'oo', 'op', 'oq', 'or', 'os', 'ot', 'ou', 'ov', 'ow', 'ox', 'oy', 'oz', 'pa', 'pb', 'pc', 'pd', 'pe', 'pf', 'pg', 'ph', 'pi', 'pj', 'pk', 'pl', 'pm', 'pn', 'po', 'pp', 'pq', 'pr', 'ps', 'pt', 'pu', 'pv', 'pw', 'px', 'py', 'pz', 'qa', 'qb', 'qc', 'qd', 'qe', 'qf', 'qg', 'qh', 'qi', 'qj', 'qk', 'ql', 'qm', 'qn', 'qo', 'qp', 'qq', 'qr', 'qs', 'qt', 'qu', 'qv', 'qw', 'qx', 'qy', 'qz', 'ra', 'rb', 'rc', 'rd', 're', 'rf', 'rg', 'rh', 'ri', 'rj', 'rk', 'rl', 'rm', 'rn', 'ro', 'rp', 'rq', 'rr', 'rs', 'rt', 'ru', 'rv', 'rw', 'rx', 'ry', 'rz', 'sa', 'sb', 'sc', 'sd', 'se', 'sf', 'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sp', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'sx', 'sy', 'sz', 'ta', 'tb', 'tc', 'td', 'te', 'tf', 'tg', 'th', 'ti', 'tj', 'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tq', 'tr', 'ts', 'tt', 'tu', 'tv', 'tw', 'tx', 'ty', 'tz', 'ua', 'ub', 'uc', 'ud', 'ue', 'uf', 'ug', 'uh', 'ui', 'uj', 'uk', 'ul', 'um', 'un', 'uo', 'up', 'uq', 'ur', 'us', 'ut', 'uu', 'uv', 'uw', 'ux', 'uy', 'uz', 'va', 'vb', 'vc', 'vd', 've', 'vf', 'vg', 'vh', 'vi', 'vj', 'vk', 'vl', 'vm', 'vn', 'vo', 'vp', 'vq', 'vr', 'vs', 'vt', 'vu', 'vv', 'vw', 'vx', 'vy', 'vz', 'wa', 'wb', 'wc', 'wd', 'we', 'wf', 'wg', 'wh', 'wi', 'wj', 'wk', 'wl', 'wm', 'wn', 'wo', 'wp', 'wq', 'wr', 'ws', 'wt', 'wu', 'wv', 'ww', 'wx', 'wy', 'wz', 'xa', 'xb', 'xc', 'xd', 'xe', 'xf', 'xg', 'xh', 'xi', 'xj', 'xk', 'xl', 'xm', 'xn', 'xo', 'xp', 'xq', 'xr', 'xs', 'xt', 'xu', 'xv', 'xw', 'xx', 'xy', 'xz', 'ya', 'yb', 'yc', 'yd', 'ye', 'yf', 'yg', 'yh', 'yi', 'yj', 'yk', 'yl', 'ym', 'yn', 'yo', 'yp', 'yq', 'yr', 'ys', 'yt', 'yu', 'yv', 'yw', 'yx', 'yy', 'yz', 'za', 'zb', 'zc', 'zd', 'ze', 'zf', 'zg', 'zh', 'zi', 'zj', 'zk', 'zl', 'zm', 'zn', 'zo', 'zp', 'zq', 'zr', 'zs', 'zt', 'zu', 'zv', 'zw', 'zx', 'zy', 'zz']


tokens.append(" @ ")
print(tokens)
mask_index = tokens.index(" @ ")
vocab_size = len(tokens)



print('vocab token size: ', vocab_size)
print('input text token size: ', len(input_tokens))


stoi = {ch:i for i,ch in enumerate(tokens)}
itos = {i:ch for i,ch in enumerate(tokens)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])





def replace_tokens(input_data, replacement_rate=0.2, replacement_token=" @ ", random_seed=None):
    """
    Replace a proportion of tokens in the input_tokens list with a replacement token.
    
    Args:
        input_tokens (list of str): List of tokens to process.
        replacement_rate (float): Fraction of tokens to replace (0 to 1).
        replacement_token (str): The token to insert as replacement.
        random_seed (int, optional): Seed for reproducibility.
    
    Returns:
        list of str: List of tokens with some replaced by the replacement token.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Determine the number of tokens to replace
    num_tokens = len(input_data)
    num_to_replace = int(num_tokens * replacement_rate)
    
    # Randomly choose indices of tokens to replace
    indices_to_replace = random.sample(range(num_tokens), num_to_replace)
    
    # Replace the selected tokens
    for idx in indices_to_replace:
        input_data[idx] = replacement_token
    
    return input_data


def replace_indexes(input_data, replacement_rate=0.001, replacement=" @ ", random_seed=None):
    """
    Replace a proportion of tokens in the input_tokens list with a replacement token.
    
    Args:
        input_tokens (list of indexes): List of tokens to process.
        replacement_rate (float): Fraction of tokens to replace (0 to 1).
        replacement (str): The value to insert as replacement.
        random_seed (int, optional): Seed for reproducibility.
    
    Returns:
        list of str: List of tokens with some replaced by the replacement token.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Determine the number of tokens to replace
    num_tokens = len(input_data)
    num_to_replace = int(num_tokens * replacement_rate)
    
    # Randomly choose indices of tokens to replace
    indices_to_replace = random.sample(range(num_tokens), num_to_replace)
    
    # Replace the selected tokens
    input_data[indices_to_replace] = replacement
    
    return input_data



data_input = torch.tensor(encode(input_tokens), dtype=torch.long)
data_target = torch.tensor(encode(input_tokens), dtype=torch.long)

data_input = replace_indexes(data_input, replacement_rate=0.001, replacement=len(tokens)-1, random_seed=random.randint(0, 2**32-1))





torch.manual_seed(135665)


def get_batch():
    #var_time = time_intervals - 2*random.randint(0, 2)
    var_time = time_intervals
    ix = torch.randint(len(data_input) - var_time, (batch_size, ))
    x = torch.stack([data_input[i:i+var_time] for i in ix])
    y = torch.stack([data_target[i+1:i+var_time+1] for i in ix])
    #y = torch.stack([data_target[i+1:i+var_time+1] for i in ix])
    return x.to(device), y.to(device)




def get_random_block():
    #var_time = time_intervals - 2*random.randint(0, 50)
    var_time = time_intervals
    i = random.randint(0, len(data_target) - var_time)
    block = data_target[i:i+var_time].reshape(1, -1).to(device)
    return block


@torch.no_grad()
def estimate_loss():
    LLM.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        _, _, loss = LLM.update(X, targets=Y, ignore_index=mask_index)
        losses[k] = loss.item()
    out = losses.mean()
    LLM.train()
    return out

def text_correct(text, multiline=True):
    def cap(match):
        return(match.group().capitalize())
    p = re.compile(r'(?<=[\.\?!][\s\n])(\w+)')
    text = text.replace(" i ", " I ")
    text = text.replace(" lord ", " Lord ")
    text = text.replace(" god ", " God ")
    for _ in range(7):
        text = text.replace(" : ", ": ")
        text = text.replace(" ! ", "! ")
        text = text.replace(" ? ", "? ")
        text = text.replace(" . ", ". ")
        text = text.replace(" , ", ", ")
    if multiline: text = text.replace(" * ", "\n")
    for _ in range(7): text = text.replace("  ", " ")


    text = p.sub(cap, text)
    return text


LLM = BigramLanguageModel(vocab_size, time_intervals, vocab_embed=vocab_embed, n_embed=n_embed, facets=facets, n_layers=layers, device=device).to(device)
optimizer = torch.optim.AdamW(LLM.parameters(), lr=learning_rate)

pytorch_total_params = sum(p.numel() for p in LLM.parameters())
print('LLM parameters: ', pytorch_total_params)

decode_in_squares = lambda l: ', '.join(['[' + itos[i] + ']' for i in l])

try:
    LLM.load_state_dict(torch.load('LLM_model.pt',  weights_only=True))
    for i in range(1):
        print("text corpus ", i, ":\n")
        context = get_random_block()
        text = decode_in_squares(LLM.generate(context, max_new_tokens=1000)[0].tolist())[-1000:]
        print(text)

    print("loaded")
except:
    print("no LLM")

for iter in range(max_iter):

    # Uncomment this if you want 0.1% tokens masked with @. One needs higher depths.
    #if iter % 50 == 0: data_input = replace_indexes(data_input, replacement_rate=0.001, replacement=len(tokens)-1, random_seed=random.randint(0, 2**32-1))
        
        

    if iter % eval_interval == 0:
        losses = estimate_loss()
        context = get_random_block()
        text = decode(LLM.generate(context, max_new_tokens=100)[0].tolist())[-100:]
        text = text_correct(text, multiline=False)

        print(f"step {iter}, train loss: {losses:.4f}, text: {text}")
        if iter>=500:
            try:
                torch.save(LLM.state_dict(), 'LLM_model.pt')
            except:
                print("problem during saving LLM")

        if iter>=10000 and iter%10000==0:
            context = get_random_block()
            text = decode(context[0].tolist())
            print(text_correct(text))
            print("###########################################")
            print("###########################################")
            text = decode(LLM.generate(context, max_new_tokens=500)[0].tolist())
            print(text_correct(text))
            print("###########################################")
            print("###########################################")

            optimizer = torch.optim.AdamW(LLM.parameters(), lr=learning_rate)

    #sample batch of data
    xb, yb = get_batch()


    optimizer.zero_grad(set_to_none=True)
    #evaluate the loss
    with torch.cuda.amp.autocast(dtype=torch.float16):
        _, _, loss = LLM.update(xb, targets=yb, ignore_index=mask_index)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    


#generate from the LLM
#context = torch.ones((1,1), dtype=torch.long, device=device)

context = get_random_block()

text = decode(context[0].tolist())
print(text_correct(text))

print("###########################################")
print("###########################################")
print("###########################################")


text = decode(LLM.generate(context, max_new_tokens=500)[0].tolist())
print(text_correct(text))



