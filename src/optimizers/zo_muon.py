class ZO_MUON(ZeroOrderOptimizer):
    def __init__(self, params, args)
        self._inner_optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, model, inputs):
        args = self.trainer.args # NOTE: call trainer. instead of self. ??? FIXED.

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None 

        self.zo_random_seed = np.random.randint(1000000000)

        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.trainer.zo_forward(model, inputs) # NOTE: call trainer. instead of self. ??? FIXED.

        if args.q == 1:
            if args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.trainer.zo_forward(model, inputs) # NOTE: call trainer. instead of self. ??? FIXED.
                self.projected_grad = ((loss1 - loss2) / args.zo_eps).item()
            else:  
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.trainer.zo_forward(model, inputs) # NOTE: call trainer. instead of self. ??? FIXED.
                self.projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()
                self.zo_perturb_parameters(scaling_factor=1)
        else:
            raise NotImplementedError("q > 1 is not implemented")

        torch.manual_seed(self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            grad_sparsity = self.trainer.get_grad_sparsity_by_name(name) # NOTE: call trainer. instead of self. ??? FIXED.
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

            if args.trainer == "zo_sign_opt":
                grad_update = np.sign(self.projected_grad) * z
            else:
                grad_update = self.projected_grad * z

            if param.ndim >= 2 and param.size(0) < 10000:
                g = grad_update.to(torch.bfloat16)
                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)  
                g_ortho = zeropower_via_newtonschulz5(g, steps=5)
                if len(original_shape) > 2:
                    g_ortho = g_ortho.view(original_shape)
                grad_update_final = g_ortho.to(param.data.dtype)
            else:
                grad_update_final = grad_update

            param.grad = grad_update_final
            self._inner_optimizer.step()
            param.grad = None

        for _, param in self.named_parameters_to_optim:
            param.grad = None

        assert args.gradient_accumulation_steps == 1

        return loss1
