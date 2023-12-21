

def SSIMLoss(prediction, target, C1=0.01, C2=0.03):

    # 计算均值和标准差
    mu_P = prediction.mean()
    mu_T = target.mean()
    sigma_P = prediction.std()
    sigma_T = target.std()
    sigma_PT = (prediction * target).mean() - mu_P * mu_T

    # 计算 SSIM 指数
    ssim_numerator = (2 * mu_P * mu_T + C1) * (2 * sigma_PT + C2)
    ssim_denominator = (mu_P**2 + mu_T**2 + C1) * (sigma_P**2 + sigma_T**2 + C2)
    ssim_index = ssim_numerator / ssim_denominator

    # 计算 SSIM 损失
    loss = 1 - ssim_index

    return loss





