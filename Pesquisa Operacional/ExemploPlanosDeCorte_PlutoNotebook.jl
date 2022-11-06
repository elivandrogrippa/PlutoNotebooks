### A Pluto.jl notebook ###
# v0.19.14

#> [frontmatter]
#> title = "Aplicação do algoritmo de planos de corte. "
#> date = "2022-11-06"
#> tags = ["PO2_amaral"]
#> description = "Uma breve aplicação do algoritmo de planos de corte a um problema de dimensionamento de lote não capacitado. "

using Markdown
using InteractiveUtils

# ╔═╡ f56bc8f5-504c-4bcd-9ec3-16236a1a0e84
using JuMP, CPLEX, Printf, MathOptInterface, PlutoUI

# ╔═╡ 8ab0430f-1bb6-4b71-be0d-2461cea061f7
md"""
# Aplicação do Algoritmo de planos de corte. 

Autores: Elivandro Oliveira Grippa e André Renato Sales Amaral. 

 - Material auxiliar para a disciplina de Pesquisa Operacional II. 

 - O código escrito na linguagem [Julia](https://julialang.org/) se baseia na implementação realizada por Leonardo Taccari, disponível no Github [leotac](https://github.com/leotac/julia-cuts).

 - Tutorial para instalação e uso da linguagem Julia (<https://leonardosecchin.github.io/julia/>).

"""

# ╔═╡ bdd7f5ae-63b4-4f00-841b-9dbaa4b182c5
md"""
## Resumo

Nesse Notebook Pluto visamos apresentar um exemplo da aplicação do algoritmo de planos de corte ao problema de dimensionamento de lote não-capacitado de item único
(do inglês _Single-item uncapacitated lot-sizing problem_), que será explicado em seções posteriores. Abordaremos uma implementação simples que pode ser melhorada e modificada de acordo com o problema alvo. 

"""

# ╔═╡ e27192d7-3627-498c-9da9-d0987c68ad67
md"""
## Problema de dimensionamento de lote não-capacitado de item único

Vamos considerar um Problema de Dimensionamento de Lote não-capacitado de item Único (PDLU), com a seguinte formulação:

$$\begin{align}
   \min\quad & \sum_{t\in T}\left(K_tz_t +  c_t q_t + h_t s_t \right) & \\
   \text{s.t.}\quad &  s_{t-1} + q_t   = d_t + s_t \qquad& \forall t \in T \\
   & 0 \le  q_t \le M z_t \qquad&\forall t \in T \\
&    s_t \ge 0 \qquad&\forall t \in T \\
&       z_t  \in\{0,1\} \qquad&\forall t \in T
\end{align}$$

Temos três variáveis chave para o período $t$: $q_t$ é o nível de produção , $z_t$ a decisão montada e  $s_t$ o nível de estoque. Para cada uma dessas variáveis há um custo associado (acrescentado na função objetivo): $c_t$, $K_t$  e $h_t$ são respectivamente, o custo de produção, custo de configuração e custo de manutenção no período $t$. O objetivo é minimizar o custo total de produção, montagem e estoque.

Este problema é simples, na verdade, polinomialmente solucionável, seja via programação dinâmica ou com uma formulação estendida.

!!! warning "Informações"
	Não será abordada a modelagem desse problema devido aos objetivos já citados. Queremos apenas ilustrar a aplicação do algoritmo de planos de corte a um problema simples com o uso da linguagem Julia. 

"""

# ╔═╡ 8cb7270c-77da-4fb2-aad9-07903d379f85
md"""
Suponha que queiramos resolvê-lo com programação inteira-mista. Nesse caso, usemos um callback do solver para adicionar alguns planos de corte para fortalecer a formulação. Especificamente, em cada nó da árvore de busca, procuramos uma desigualdade violada do conjunto de desigualdades $(l,S)$:

$$\begin{equation}
\sum_{j\in S}q_i \leq \sum_{j\in S}d_{jl}z_{j} + s_{l},\qquad \forall l\in T, \forall S\subseteq\{1,\dots,l\}
\end{equation},$$
que será adicionado à formulação por meio de um retorno de chamada _UserCut_.
"""

# ╔═╡ 34a6e976-c20c-4881-83fb-a3b4df40d3f6
md"""
## Implementação do modelo
"""

# ╔═╡ 50818b78-830d-4ff0-9767-119ad578a9ee
md"""
#### Pacotes utilizados

* [JuMP](https://github.com/jump-dev/JuMP.jl): Linguagem de modelagem;
* [CPLEX](https://github.com/jump-dev/CPLEX.jl): Solver IBM CPLEX;
* [GLPK](https://github.com/jump-dev/GLPK.jl): Kit de programação linear GNU;
* Printf : Pacote que incorpora o printf da linguagem C;
* [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl): Camada de abstração para solucionadores de otimização matemática;
* [PlutoUI](https://github.com/JuliaPluto/PlutoUI.jl): Ferramentas para o Notebook Pluto.

"""

# ╔═╡ 9fc967ae-fb9c-4fbc-ae86-6325f0666f3c
md"""
#### Funções auxiliares

Destinadas a leitura de instâncias do problema (informações fornecidas) e separação para inclusão dos cortes. 
"""

# ╔═╡ b47e029b-960b-4897-9ad7-e22ff32a4023
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

# ╔═╡ 05358c29-aa35-4226-a75d-975a700afca8
md"""
!!! warning "O que é um callback?"
	Uma função callback é uma função passada a outra função como argumento, que é então invocado dentro da função externa para completar algum tipo de rotina ou ação.
"""

# ╔═╡ 99cb3c67-1015-4302-976f-80033fb524a4
md"""
A função abaixo (_separate_) nos informa quais desigualdade $(l, S)$ são violadas. Tornando necessária (ou não, de acordo com o solver) a adiciona-la no modelo e reotimizar. Caso esse corte seja inviável, com o uso do callback teremos um apontamento do erro por parte do solver. Portanto, é necessário avaliar se os cortes a serem inseridos sejam cortes válidos, externo a envoltória convexa.

Uma explicação breve sobre essa rotina pode ser encontrada [_aqui_](https://leotac.github.io/posts/2015/06/10/julia/#fnref:1).
"""

# ╔═╡ d3c72b17-75b7-4e3a-9174-e8b2c38291a4
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

# ╔═╡ 71a00cd7-b049-4dee-82c1-8c98c1e6294f
md"""
#### Construção do modelo

Nessa seção apresentarei a construção do modelo, com o uso do JuMP, e sua resolução com o uso de um solver (no caso, o CPLEX ou GLPK).

"""

# ╔═╡ 1c36eacf-1946-4817-b080-f3e2a7c8866f
md"""
###### Dados de entrada:

* _path_: Diretório da instância a ser utilizada;
* _valid_: Booleano para a insersão de cortes;
* _solver_: CPLEX.Optimizer (exige licença) ou GLPK.Optmizer (open-source).

###### Dados de saída:

* _Solution Summary_: Informações gerais do solver;
* _objective-value_ : Valor ótimo;
*  $z_t$ : Solução referente a decisão montada;
*  $q_t$ : Solução referente ao nível de produção;
*  $s_t$ : Solução referente ao nível de estoque;
* _separation_ : Número de separações;
* _separationtime_ : Tempo de separação;
* _NumberOfUseCuts_ : Número de cortes inseridos (ou _UserCuts_).
"""

# ╔═╡ 3c523c4f-3b52-46b5-94a2-2b261bd87c02
md"""
A estrutura abaixo facilita a organização dos dados de retorno da função principal (contrução do modelo + resolução).
"""

# ╔═╡ 5f3f90b2-89b8-4f72-93a5-01b0c42ce146
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

# ╔═╡ 7d799d23-cae3-495a-91dd-1b16ff307af0
md"""
##### Função principal
"""

# ╔═╡ 6f5408a2-331c-40a2-8534-b675aa87e494
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

# ╔═╡ aee862d7-df4e-407c-a7b1-796981500810
md"""
##### Chamada da função principal
"""

# ╔═╡ 07e89b59-d592-48ac-995a-1e3c252ca620
solution_data = solvePDLU("test.dat", valid = true);

# ╔═╡ 77b91982-c8ae-4186-996a-9cfd3c82248d
solution_data.solution_summary

# ╔═╡ 972e9a42-458f-4f92-a402-79fd23aa4d28
md"""
##### Links auxiliares
* [JuMP](https://jump.dev/JuMP.jl/stable/)
* [MathOptInterface](https://jump.dev/MathOptInterface.jl/dev/)
"""

# ╔═╡ 18f41d15-6638-4032-8e04-010dee10d3df
PlutoUI.TableOfContents(title = "Sumário")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CPLEX = "a076750e-1247-5638-91d2-ce28b192dca0"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
CPLEX = "~0.9.4"
JuMP = "~1.4.0"
MathOptInterface = "~1.9.0"
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "1408deffa128384d39941c35faa23ed3355fa654"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPLEX]]
deps = ["CEnum", "Libdl", "MathOptInterface", "SparseArrays"]
git-tree-sha1 = "32357584cf030fce4b5e2e1852505619283a9d25"
uuid = "a076750e-1247-5638-91d2-ce28b192dca0"
version = "0.9.4"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "8b7a4d23e22f5d44883671da70865ca98f2ebf9d"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays"]
git-tree-sha1 = "9a57156b97ed7821493c9c0a65f5b72710b38cf7"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.4.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "ceed48edffe0325a6e9ea00ecf3607af5089c413"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.9.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "1d57a7dc42d563ad6b5e95d7a8aebd550e5162c0"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.5"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "cceb0257b662528ecdf0b4b4302eb00e767b38e7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─8ab0430f-1bb6-4b71-be0d-2461cea061f7
# ╟─bdd7f5ae-63b4-4f00-841b-9dbaa4b182c5
# ╟─e27192d7-3627-498c-9da9-d0987c68ad67
# ╟─8cb7270c-77da-4fb2-aad9-07903d379f85
# ╟─34a6e976-c20c-4881-83fb-a3b4df40d3f6
# ╟─50818b78-830d-4ff0-9767-119ad578a9ee
# ╠═f56bc8f5-504c-4bcd-9ec3-16236a1a0e84
# ╟─9fc967ae-fb9c-4fbc-ae86-6325f0666f3c
# ╠═b47e029b-960b-4897-9ad7-e22ff32a4023
# ╟─05358c29-aa35-4226-a75d-975a700afca8
# ╟─99cb3c67-1015-4302-976f-80033fb524a4
# ╠═d3c72b17-75b7-4e3a-9174-e8b2c38291a4
# ╟─71a00cd7-b049-4dee-82c1-8c98c1e6294f
# ╟─1c36eacf-1946-4817-b080-f3e2a7c8866f
# ╟─3c523c4f-3b52-46b5-94a2-2b261bd87c02
# ╠═5f3f90b2-89b8-4f72-93a5-01b0c42ce146
# ╟─7d799d23-cae3-495a-91dd-1b16ff307af0
# ╠═6f5408a2-331c-40a2-8534-b675aa87e494
# ╟─aee862d7-df4e-407c-a7b1-796981500810
# ╠═07e89b59-d592-48ac-995a-1e3c252ca620
# ╠═77b91982-c8ae-4186-996a-9cfd3c82248d
# ╟─972e9a42-458f-4f92-a402-79fd23aa4d28
# ╟─18f41d15-6638-4032-8e04-010dee10d3df
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
