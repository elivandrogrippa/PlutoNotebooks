using JuMP, CPLEX, Printf, MathOptInterface


struct PDLU_Info
	solution_summary
	objective_value
	z_t
	q_t
	s_t
	separation
	separationtime
	NumberOfUseCuts # Obs.: Não consegui obter de maneira automática
end


function readPDLU(path::String)
	# Abrindo o arquivo da intância
	f = open(path)
	
	# Obtendo as informações da instância
	T = parse(Int64, readline(f))
	c = parse.(Float64,split(strip(readline(f))))
	h = parse.(Float64,split(strip(readline(f))))
	K = parse.(Float64,split(strip(readline(f))))
	d = parse.(Float64,split(strip(readline(f))))
	
	# Fechando o arquivo path
	close(f)
	
	# Retornando informações da intância
	return T, c, h, K, d
end

begin
	const TOL = 1e-6
	
	function separate(T::Int64, sumd::Array{Float64, 2}, z_val, q_val, z, q)
		S = zeros(Bool, T)
		for l in 1:T
			fill!(S, false)
			lhsvalue = 0.  #q(L\S) + sum{d[j:l]*z[j] for j in S}
			empty = true
			for j in 1:l
				if q_val[j] > sumd[j,l]*z_val[j] + TOL
					S[j] = true
					empty = false
					lhsvalue += sumd[j,l]*z_val[j]
				else
					lhsvalue += q_val[j]
				end
			end
			if empty
				continue
			end
			if lhsvalue < sumd[1,l] - TOL
				lhs = sum(q[1:l])
				for j = (1:T)[S]
					lhs += sumd[j,l]*z[j] - q[j]
				end
				return lhs - sumd[1,l]
			end
		end
		return nothing
	end
end

function solvePDLU(path::String; solver = CPLEX.Optimizer, valid::Bool = true)

	# Solver CPLEX (https://github.com/jump-dev/CPLEX.jl)
	# solver GLPK (https://github.com/jump-dev/GLPK.jl)

    # Leitura das informações no problema test path.
	T, c, h, K, d = readPDLU(path)

    # Inicialização do modelo, chamaremos de modelo m
	m = Model(solver)
	
	# set_silent(m)
	
	# Parâmetros carregados para o CPLEX: 
	# - Desliga os cortes automáticos do CPLEX
	set_optimizer_attribute(m, "CPX_PARAM_CUTSFACTOR", 1)

    # Definindo as variáveis do modelo m
	@variable(m, z[1:T], Bin)
	@variable(m, q[i = 1:T] >= 0)
	@variable(m, s[1:T] >= 0)

    # Construção da função objetivo
	@objective(m, Min, sum(K[t]*z[t] + c[t]*q[t] +  h[t]*s[t] for t in 1:T))

    # Adicionando as restrições iniciais do modelo m
	@constraint(m, balance[t = 1:T], (t>1 ? s[t-1] : 0) + q[t] == d[t] + s[t])
	@constraint(m, activation[t = 1:T], q[t] <= sum(d[t:T])*z[t])
	
	# Calcule previamente sum(d[j:l]) para o corte
	sumd = zeros(Float64, T, T)
	for l = 1:T, j = 1:l
	   sumd[j,l] = sum(d[j:l])
	end

	# Tempo de separação, número de separação e chamadas
	separationtime = 0.
	separations = 0
	called = 0

	# função que nos fornece o corte a ser utilizado (callback)
	function lSgenerator(cb_data)
		called += 1
		tt = time()

		# Retorna a solução primal de uma variável dentro de um callback
		z_val = callback_value.((cb_data,), m[:z])
		q_val = callback_value.((cb_data,), m[:q])
		
		# Função que nos retorna a desigualdade quando correta quando violada. 
		expr = separate(T, sumd, z_val, q_val, z, q)

		# carrega a restrição e envia para o modelo
		if expr != nothing
			con =  @build_constraint(expr >= 0)
			MOI.submit(m, MOI.UserCut(cb_data), con)
		end

		# Contadores
		separationtime += time() - tt
		separations += 1
	end 

	# Inclusão do corte se valid == true (controle adequado)
	if valid
		MOI.set(m, MOI.UserCutCallback(), lSgenerator)
	end

	# Resolvendo o modelo m
	status = optimize!(m)

	# Imprimindo na tela as informações referente a solução. 
	@printf("Objective value: %.2f\n", objective_value(m))
	@printf("Separation time: %.2f ms\n", separationtime*1000)
	println("Separated: $separations")
	
	return PDLU_Info(solution_summary(m, verbose=false), objective_value(m), JuMP.value.(z), JuMP.value.(q), JuMP.value.(s), separations, separationtime*1000, nothing)
end 

solution_data = solvePDLU("test.dat", valid = true);

solution_data.solution_summary


