# 📘 PINN Study

Raissi et al. (2019) *Physics-Informed Neural Networks* 논문을 PyTorch로 직접 구현하며 공부하는 저장소.

**논문:** [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

## continuous model

### Section 3.1. data-driven solution of partial differential equations
#### Schrödinger (Forward Problem)
- 네트워크 입력: `(t, x)` → 출력: `(u, v)` (실수부, 허수부)
- 손실함수: `MSE = MSE_0 + MSE_b + MSE_f`
- Collocation point 20,000개로 PDE residual 계산
- Optimizer: L-BFGS
### Section 4.1. Data-driven discovery of partial differential equations
#### Navier-Stokes (Inverse Problem)
- 네트워크 입력: `(t, x, y)` → 출력: `(ψ, p)`
- `u = ψ_y`, `v = -ψ_x` 로 연속 방정식 자동 만족
- 미지 파라미터 `λ₁`, `λ₂`를 `nn.Parameter`로 학습
- Optimizer: Adam → L-BFGS 2단계 학습

