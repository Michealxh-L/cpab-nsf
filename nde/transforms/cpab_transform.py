import torch

eps = torch.finfo(torch.float32).eps


def cmpf0(x):
    return torch.abs(x) < eps


def batch_effect(x, theta):
    if x.ndim == 1:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]
        x = torch.broadcast_to(x, (n_batch, n_points))  # .flatten()
    return x.flatten()


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
    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = cmpf0(a)
    x1 = x + t * b
    eta = torch.exp(t * a)
    x2 = eta * x + (b / a) * (eta - 1.0)
    # x2 = torch.exp(t * a) * (x + (b / a)) - (b / a)
    psi = torch.where(cond, x1, x2)
    return psi


def get_hit_time(x, theta, params):
    thit = torch.empty_like(x)
    valid = torch.full_like(x, True, dtype=bool)

    c = get_cell(x, params)
    A, r = get_affine(x, theta, params)

    a = A[r, c, 0]
    b = A[r, c, 1]

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
    thit[~cond] = torch.where(
        cmpf0(a[~cond]), (xc[~cond] - x[~cond]) / b[~cond], torch.log(vc[~cond] / v[~cond]) / a[~cond],
    )
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

    while True:
        thit, xc, cc, a = get_hit_time(x, theta, params)
        psi = get_psi(x, t, theta, params)

        valid = thit > t
        phi[~done] = psi

        valid_t = thit < time
        t_temp = None
        if torch.all(valid):
            t_temp = t
        else:
            t_temp = thit
            t_temp[~valid_t] = time

        log_jac[~done] += a * t_temp
        done[~done] = valid

        if torch.all(valid):
            return phi.reshape((n_batch, -1)), log_jac

        params.r = params.r[~valid]
        x = xc[~valid]
        c = cc[~valid]
        t = (t - thit)[~valid]

        cont += 1
        nc = params.nc
        # nc = 20
        if cont > nc: # print the cell index and check
            raise BaseException
    return None


def cpab_transform(T, input, theta, time=1.0):
    params = T.params
    output, log_jac = integrate_closed_form(input, theta, params, time)
    return output, log_jac