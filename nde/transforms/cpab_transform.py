import torch

eps = torch.finfo(torch.float32).eps


def cmpf0(x):
    return torch.abs(x) < eps


def batch_effect(x, theta):
    if x.ndim == 1:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]
        x = torch.broadcast_to(x, (n_batch, n_points))
    shape = x.shape
    return x.flatten().reshape(shape[0],shape[-1])


def get_affine(x, theta, params):
    if params.precomputed:
        return params.A, params.r
    else:
        n_batch = theta.shape[0]
        dim = theta.shape[-1]
        theta = theta.reshape(n_batch, dim)
        n_points = x.shape[-1]
        repeat = int(n_points / n_batch)
        # r = np.broadcast_to(np.arange(n_batch), [n_points, n_batch]).T
        # NOTE: here we suppose batch effect has been already executed
        r = torch.arange(n_batch).repeat_interleave(repeat).long().to(x.device)
        B = params.B.to(x.device)
        # print(B.shape,theta.shape,n_batch)
        A = B.mm(theta.T).T.reshape(n_batch, -1, 2).to(x.device)

        return A, r


def precompute_affine(xs, thetas, params):
    params = params.copy()
    params.precomputed = False
    params.A, params.r = get_affine(xs, thetas, params)
    params.precomputed = True
    return params


def right_boundary(c, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc
    return xmin + (c + 1) * (xmax - xmin) / nc + eps


def left_boundary(c, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc
    return xmin + c * (xmax - xmin) / nc - eps


def get_cell(x, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc

    c = torch.floor((x - xmin) / (xmax - xmin) * nc)
    c = torch.clamp(c, 0, nc - 1).long()
    return c


def get_psi(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    
    shape = A.shape
    num = shape[0]

    A = A.reshape([num,shape[-2],shape[-1]])
    idx = torch.cat((c,c),1).reshape(c.shape[0],1,2)
    a = A.gather(1,idx).reshape(num,2,1)[:,0]
    b = A.gather(1,idx).reshape(num,2,1)[:,1]

    cond = cmpf0(a)
    x1 = x + t * b
    eta = torch.exp(t * a)
    x2 = eta * x + (b / a) * (eta - 1.0)
    # x2 = torch.exp(t * a) * (x + (b / a)) - (b / a)
    psi = torch.where(cond, x1, x2)
    return psi


def get_hit_time(x, theta, params):
    thit = torch.empty_like(x)

    c = get_cell(x, params)
    A, r = get_affine(x, theta, params)
    # A = A[~valid]
    shape = A.shape
    num = shape[0]
    A = A.reshape([num,shape[-2],shape[-1]])
    idx = torch.cat((c,c),1).reshape(c.shape[0],1,2)
    a = A.gather(1,idx).reshape(num,2,1)[:,0]
    b = A.gather(1,idx).reshape(num,2,1)[:,1]

    v = a * x + b
    cond1 = cmpf0(v)

    cc = c + torch.sign(v)
    cond2 = torch.logical_or(cc < 0, cc >= params.nc)
    xc = torch.where(v > 0, right_boundary(c, params), left_boundary(c, params))

    vc = a * xc + b
    cond3 = cmpf0(vc)
    cond4 = torch.sign(v) != torch.sign(vc)
    cond5 = torch.logical_or(xc == params.xmin, xc == params.xmax)

    cond = cond1 | cond2 | cond3 | cond4 | cond5
    #thit[~cond] = torch.where(cmpf0(a[~cond]), (xc[~cond] - x[~cond]) / b[~cond], torch.log(vc[~cond] / v[~cond]) / a[~cond])
    b0_index = abs(b[~cond])<5e-16
    zero_index = (xc[~cond] - x[~cond]) == 0
    intersect_index = b[~cond]==0 ### intersection of b0_index and zero_index
    intersect_index = intersect_index & zero_index
    div1 = torch.empty_like(b[~cond])
    div1[b0_index] = float("inf")
    div1[zero_index] = float(0)
    div1[~b0_index] = (xc[~cond][~b0_index] - x[~cond][~b0_index]) / b[~cond][~b0_index] ### if xc-x is 0, the result should be 0 rather inf
    div1[~intersect_index] = float("inf")
    # div1 = (xc[~cond] - x[~cond]) / b[~cond]  ### b is too small that cause the inifite issue
    div_temp = torch.log(vc[~cond] / v[~cond]) # vc and v is nagative, which cause nan by using log, to avoid this we need alternatively apply div1
    div2 = (div_temp) / (a[~cond])
    thit[~cond] = torch.where(cmpf0(a[~cond]), div1, div2)
    thit[cond] = float("inf")
    return thit, xc, cc, a


def integrate_closed_form(x, theta, params, time=1.0):
    # setup
    x = batch_effect(x, theta)
    t = torch.ones_like(x) * time
    params = precompute_affine(x, theta, params)
    n_batch = theta.shape[0]

    # computation
    phi = torch.empty_like(x)
    done = torch.full_like(x, False, dtype=bool)

    c = get_cell(x, params)  # the cell index of x
    cont = 0
    

    log_jac = torch.zeros_like(x)
    h_valid = torch.full_like(x, False, dtype=bool)

    while True:
        thit, xc, cc, a = get_hit_time(x, theta, params)
        psi = get_psi(x, t, theta, params)

        valid = thit > t
        valid_t = thit < time
        phi[~done] = psi[~h_valid]
    
        t_temp = None

        if torch.all(valid):
            t_temp = t
        else:
            t_temp = thit
            t_temp[~valid_t] = time
            # t_temp = torch.nan_to_num(thit.clone()*valid_t) + torch.full([n_batch,1],time)*(~valid_t)
            # thit = t_temp.clone()
        add = a*t_temp
        log_jac = log_jac*done + log_jac*(~done)+add*(~h_valid)
        done = done*(done.clone()) + valid*(~h_valid)
        h_valid[done] = True

        if torch.all(valid):
            return phi.reshape((n_batch, -1)), log_jac

        x = x*valid + xc*(~valid)
        shape = x.shape
        x = x.reshape(shape[0],1)
        t = t*valid + (t.clone() - thit)*(~valid)
        shape = t.shape
        t = t.reshape(shape[0],1)

        cont += 1
        nc = params.nc
        if cont > nc:
            raise BaseException
    return None


def cpab_transform(T, input, theta, time=1.0):
    params = T.params
    output, log_jac = integrate_closed_form(input, theta, params, time)
    return output, log_jac
