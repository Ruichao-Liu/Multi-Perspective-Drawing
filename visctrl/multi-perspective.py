def ddcm_sampler(scheduler, x_s, x_t, timestep, e_s, e_t, x_0, noise, eta, to_next=True):
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    prev_step_index = scheduler.step_index + 1
    if prev_step_index < len(scheduler.timesteps):
        prev_timestep = scheduler.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev
    std_dev_t = eta * variance
    noise = std_dev_t ** (0.5) * noise

    e_c = (x_s - alpha_prod_t ** (0.5) * x_0) / (1 - alpha_prod_t) ** (0.5)

    pred_x0 = x_0 + ((x_t - x_s) - beta_prod_t ** (0.5) * (e_t - e_s)) / alpha_prod_t ** (0.5)
    eps = (e_t - e_s) + e_c
    dir_xt = (beta_prod_t_prev - std_dev_t) ** (0.5) * eps

    # Noise is not used for one-step sampling.
    if len(scheduler.timesteps) > 1:
        prev_xt = alpha_prod_t_prev ** (0.5) * pred_x0 + dir_xt + noise
        prev_xs = alpha_prod_t_prev ** (0.5) * x_0 + dir_xt + noise
    else:
        prev_xt = pred_x0
        prev_xs = x_0

    if to_next:
        scheduler._step_index += 1
    return prev_xs, prev_xt, pred_x0  # 更新后的xs、xt、x0

