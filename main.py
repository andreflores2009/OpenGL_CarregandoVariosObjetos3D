# Luzes

# Neste exemplo, especificamos uma iluminação ambiente, difusa e especular básicas.
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from object_loader import ObjLoader
from camera import Camera
import pyrr

Window = None
Shader_program = None
shader_luz = None

vao_retangulo = None

#Objetos que foram baixados e estao na pasta 'meshes' e 'textures'
#Devem ser preenchidos conforme o nome que esta na pasta neste exemplo estava configurado para um objeto chamado 'pc.obj' e 'pc.png'.....
objetos = [

    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/pc.obj',
        "textura": 'textures/pc.png',
        "transformacao": [0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/sala.obj',
        "textura": 'textures/sala.png',
        "transformacao": [1, 0, 1.2, 3, 3, 3, 0, 90, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/dara.obj',
        "textura": 'textures/dara.png',
        "transformacao": [1, 0, 1.2, 1, 1, 1, 0, 90, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/audio.obj',
        "textura": 'textures/audio.png',
        "transformacao": [5, 0, 0, 3, 3, 3, 0, 0, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/tank.obj',
        "textura": 'textures/tank.dds',
        "transformacao": [12, 0, 0, 2, 2, 2, 0, 0, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/tirex.obj',
        "textura": 'textures/tirex.jpg',
        "transformacao": [-200, 100, 50, 0.1, 0.1, 0.1, 0, -45, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/home_de_ferro.obj',
        "textura": 'textures/audio.png',
        "transformacao": [1, 0, 1.2, 0.02, 0.02, 0.02, 0, 90, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/macaco.obj',
        "textura": 'textures/macaco.jpg',
        "transformacao": [-120, -190, 50, 0.02, 0.02, 0.02, 90, 0, 0]
    },
    {
        "vao_objeto": None,
        "obj_indices": None,
        "obj_buffer": None,
        "obj_textura": None,
        "mesh": 'meshes/ak.obj',
        "textura": 'textures/ak.jpg',
        "transformacao": [-65, -52, 75, 0.05, 0.05, 0.05, 0, 0, 90]
    },

]

WIDTH = 800
HEIGHT = 600

Tempo_entre_frames = 0  # variavel utilizada para movimentar a camera

# Variáveis referentes a luz
luz_posicao = pyrr.Vector3([0.0, 0.0, 0.0])
La = pyrr.Vector3([0.2, 0.2, 0.2])  # Luz ambiente
Ld = pyrr.Vector3([0.7, 0.7, 0.7])  # cinza bem claro
Ls = pyrr.Vector3([1.0, 1.0, 1.0])  # luz branca
luz_speed = 50.0  # velocidade da luz qdo movimentada pelo teclado, 1 unidade por segundo

renderiza_luz = True
luz_2_posicao = pyrr.Vector3([0.0, 0.0, 0.0])
La2 = pyrr.Vector3([0.2, 0.2, 0.2])  # Luz ambiente
Ld2 = pyrr.Vector3([1, 0, 0])  # cinza bem claro
Ls2 = pyrr.Vector3([1.0, 1, 1])  # luz branca
luz_2_speed = 50.0  # velocidade da luz qdo movimentada pelo teclado, 1 unidade por segundo

cam = Camera()

lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward = False, False, False, False


def redimensionaCallback(window, w, h):
    global WIDTH, HEIGHT
    WIDTH = w
    HEIGHT = h


def tecladoCallback(window, key, scancode, action, mode):
    global left, right, forward, backward, Tempo_entre_frames, luz_posicao, renderiza_luz
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    # if key in [glfw.KEY_W, glfw.KEY_S, glfw.KEY_D, glfw.KEY_A] and action == glfw.RELEASE:
    #     left, right, forward, backward = False, False, False, False

    # parâmetros da posição da luz
    if glfw.PRESS == glfw.get_key(Window, glfw.KEY_KP_4):
        luz_posicao[0] -= luz_speed * Tempo_entre_frames
    if glfw.PRESS == glfw.get_key(Window, glfw.KEY_KP_6):
        luz_posicao[0] += luz_speed * Tempo_entre_frames

    if glfw.PRESS == glfw.get_key(Window, glfw.KEY_KP_5):
        luz_posicao[1] -= luz_speed * Tempo_entre_frames
    if glfw.PRESS == glfw.get_key(Window, glfw.KEY_KP_8):
        luz_posicao[1] += luz_speed * Tempo_entre_frames

    if glfw.PRESS == glfw.get_key(Window, glfw.KEY_KP_7):
        luz_posicao[2] -= luz_speed * Tempo_entre_frames
    if glfw.PRESS == glfw.get_key(Window, glfw.KEY_KP_9):
        luz_posicao[2] += luz_speed * Tempo_entre_frames

    if glfw.PRESS == glfw.get_key(Window, glfw.KEY_F):
        renderiza_luz = not renderiza_luz

    print(luz_posicao)

def inicializa_retangulo():
    global vao_retangulo
    # Vao do cubo
    vao_retangulo = glGenVertexArrays(1)
    glBindVertexArray(vao_retangulo)

    # VBO dos vértices do cubo
    # VBO dos vértices do quadrado
    points = [
        # face frontal
        0.5, 0.5, 1,  # 0
        0.5, -0.5, 1,  # 1
        -0.5, -0.5, 1,  # 2
        -0.5, 0.5, 1,  # 3
        0.5, 0.5, 1,
        -0.5, -0.5, 1,
        # face trazeira
        0.5, 0.5, -1,  # 4
        0.5, -0.5, -1,  # 5
        -0.5, -0.5, -1,  # 6
        -0.5, 0.5, -1,  # 7
        0.5, 0.5, -1,
        -0.5, -0.5, -1,
        # face esquerda
        -0.5, -0.5, 1,
        -0.5, 0.5, 1,
        -0.5, -0.5, -1,
        -0.5, -0.5, -1,
        -0.5, 0.5, -1,
        -0.5, 0.5, 1,
        # face direita
        0.5, -0.5, 1,
        0.5, 0.5, 1,
        0.5, -0.5, -1,
        0.5, -0.5, -1,
        0.5, 0.5, -1,
        0.5, 0.5, 1,
        # face baixo
        -0.5, -0.5, 1,
        0.5, -0.5, 1,
        0.5, -0.5, -1,
        0.5, -0.5, -1,
        -0.5, -0.5, -1,
        -0.5, -0.5, 1,
        # face cima
        -0.5, 0.5, 1,
        0.5, 0.5, 1,
        0.5, 0.5, -1,
        0.5, 0.5, -1,
        -0.5, 0.5, -1,
        -0.5, 0.5, 1,
    ]
    points = np.array(points, dtype=np.float32)
    pvbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, pvbo)
    glBufferData(GL_ARRAY_BUFFER, points, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    # VBO das normais
    normais = [
        # face frontal
        0.0, 0.0, 1.0,  # apontado para frente
        0.0, 0.0, 1.0,  # apontado para frente
        0.0, 0.0, 1.0,  # apontado para frente
        0.0, 0.0, 1.0,  # apontado para frente
        0.0, 0.0, 1.0,  # apontado para frente
        0.0, 0.0, 1.0,  # apontado para frente
        # face trazeira
        0.0, 0.0, -1.0,  # apontado para trás
        0.0, 0.0, -1.0,  # apontado para trás
        0.0, 0.0, -1.0,  # apontado para trás
        0.0, 0.0, -1.0,  # apontado para trás
        0.0, 0.0, -1.0,  # apontado para trás
        0.0, 0.0, -1.0,  # apontado para trás
        # face esquerda
        -1.0, 0.0, 0.0,  # apontado para esquerda e para frente
        -1.0, 0.0, 0.0,  # apontado para esquerda e para frente
        -1.0, 0.0, 0.0,  # apontado para esquerda e para frente
        -1.0, 0.0, 0.0,  # apontado para esquerda e para frente
        -1.0, 0.0, 0.0,  # apontado para esquerda e para frente
        -1.0, 0.0, 0.0,  # apontado para esquerda e para frente
        # face direita
        1.0, 0.0, 0.0,  # apontado para direita e para frente
        1.0, 0.0, 0.0,  # apontado para direita e para frente
        1.0, 0.0, 0.0,  # apontado para direita e para frente
        1.0, 0.0, 0.0,  # apontado para direita e para frente
        1.0, 0.0, 0.0,  # apontado para direita e para frente
        1.0, 0.0, 0.0,  # apontado para direita e para frente
        # face inferior
        0.0, -1.0, 0.0,  # apontado para baixo e para frente
        0.0, -1.0, 0.0,  # apontado para baixo e para frente
        0.0, -1.0, 0.0,  # apontado para baixo e para frente
        0.0, -1.0, 0.0,  # apontado para baixo e para frente
        0.0, -1.0, 0.0,  # apontado para baixo e para frente
        0.0, -1.0, 0.0,  # apontado para baixo e para frente
        # face superior
        0.0, 1.0, 0.0,  # apontado para cima e para frente
        0.0, 1.0, 0.0,  # apontado para cima e para frente
        0.0, 1.0, 0.0,  # apontado para cima e para frente
        0.0, 1.0, 0.0,  # apontado para cima e para frente
        0.0, 1.0, 0.0,  # apontado para cima e para frente
        0.0, 1.0, 0.0,  # apontado para cima e para frente
    ]
    normais = np.array(normais, dtype=np.float32)
    nvbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, nvbo)
    glBufferData(GL_ARRAY_BUFFER, normais, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

def mouseCallback(window, xpos, ypos):
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    cam.process_mouse_movement(xoffset, yoffset)


def movimentaCamera():
    if left:
        cam.process_keyboard("LEFT", 0.05)
    if right:
        cam.process_keyboard("RIGHT", 0.05)
    if forward:
        cam.process_keyboard("FORWARD", 0.4)
    if backward:
        cam.process_keyboard("BACKWARD", 0.05)


def inicializaOpenGL():
    global Window, WIDTH, HEIGHT

    # Inicializa GLFW
    glfw.init()

    # Criação de uma janela
    Window = glfw.create_window(WIDTH, HEIGHT, "Quarto do madara", None, None)
    if not Window:
        glfw.terminate()
        exit()

    glfw.set_window_size_callback(Window, redimensionaCallback)  # função callback de redimensionamento
    glfw.set_key_callback(Window, tecladoCallback)  # função callback do teclado
    glfw.set_cursor_pos_callback(Window, mouseCallback)  # função callback do mouse
    glfw.set_input_mode(Window, glfw.CURSOR, glfw.CURSOR_DISABLED)  # desabilita o cursor
    glfw.make_context_current(Window)

    print("Placa de vídeo: ", OpenGL.GL.glGetString(OpenGL.GL.GL_RENDERER))
    print("Versão do OpenGL: ", OpenGL.GL.glGetString(OpenGL.GL.GL_VERSION))


def inicializaObjetos():
    global objetos

    for objeto in objetos:
        objeto['obj_indices'], objeto['obj_buffer'] = ObjLoader.load_model(objeto['mesh'])
        # Vao do objeto
        objeto['vao_objeto'] = glGenVertexArrays(1)
        glBindVertexArray(objeto['vao_objeto'])

        # VBO do objeto. Ao invés de termos 1 VBO para cada informação (vértices, texturas, normais, etc.),
        # este exemplo utiliza 1 único VBO com todas as informações dentro
        bvbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bvbo)
        glBufferData(GL_ARRAY_BUFFER, objeto['obj_buffer'].nbytes, objeto['obj_buffer'], GL_STATIC_DRAW)
        # vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, objeto['obj_buffer'].itemsize * 8, ctypes.c_void_p(0))
        # texturas
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, objeto['obj_buffer'].itemsize * 8, ctypes.c_void_p(12))
        # normais
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, objeto['obj_buffer'].itemsize * 8, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        objeto['obj_textura'] = glGenTextures(1)
        objeto['tex_w'], objeto['tex_h'], objeto['tex_data'] = ObjLoader.load_texture(objeto['textura'],
                                                                                      objeto['obj_textura'])


def inicializa_shaders():
    global Shader_program
    # Especificação do Vertex Shader:
    vertex_shader = """
        #version 400
        layout(location = 0) in vec3 vertex_posicao; //vértices do objeto vindas do modelo do objeto (PYTHON)
        layout(location = 1) in vec2 vertex_texture; //normais do objeto vindas do modelo do objeto (PYTHON)
        layout(location = 2) in vec3 vertex_normal; //normais do objeto vindas do modelo do objeto (PYTHON)
        out vec3 vertex_posicao_cam, vertex_normal_cam;
        out vec2 v_texture; //saida da textura para o fragment shader
        uniform mat4 transform, view, proj;
        void main () {
            vertex_posicao_cam = vec3 (view * transform * vec4 (vertex_posicao, 1.0)); //posição do objeto em relação a CÂMERA
            vertex_normal_cam = vec3 (view *  transform * vec4 (vertex_normal, 0.0)); //normais do objeto em relação a CÂMERA

            v_texture = vertex_texture;

            gl_Position = proj * view * transform * vec4 (vertex_posicao, 1.0);
        }
    """
    vs = OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(vs, 512, None)
        print("Erro no vertex shader:\n", info_log)

    # Especificação do Fragment Shader:
    fragment_shader = """
        #version 400
        in vec3 vertex_posicao_cam, vertex_normal_cam; //variáveis vindas do VERTEX SHADER
        in vec2 v_texture; //variável de textura vinda do VERTEX SHADER
        uniform sampler2D s_texture;
        
        //propriedades de uma luz pontual vindas do PYTHON
        uniform vec3 luz_posicao;
        uniform vec3 Ls;// luz especular
        uniform vec3 Ld;// luz difusa
        uniform vec3 La;// luz ambiente

        //propriedades de reflexão da superficie do objeto vindas do PYTHON
        uniform vec3 Ks;//reflexão especular
        uniform vec3 Kd;//reflexão difusa
        uniform vec3 Ka;//reflexão ambiente
        uniform float especular_exp;//expoente especular
        
        uniform mat4 view; //matriz da câmera vinda do PYTHON
        out vec4 frag_colour;

        //variáveis globais que são utilizadas tanto na intensidade difusa quanto especular, para não precisar recalcular duas vezes
        vec3 luz_posicao_cam, luz_vetor_cam, luz_vetor_cam_normalizada, vertex_normal_cam_normalizada;

        vec3 intensidadeAmbiente(){
            /*
            Cálculo de Intensidade de Luz Ambiente (Ia)
            O cálculo da intensidade de luz ambiente é o mais simples:
            basta multiplicar a cor da luz ambiente pela refletância de luz ambiente da superfície
            */
            vec3 Ia = La * Ka;
            return Ia;
        }

        vec3 intensidadeDifusa(){
            /*
            Cálculo de Intensidade de Luz Difusa (Id)
            Para calcularmos a intensidade de luz difusa, precisamos, primeiramente, 
            calcular a posição da luz em relação a câmera (luz_posicao_cam)
            */
            luz_posicao_cam = vec3 (view * vec4 (luz_posicao, 1.0));//posicao da luz em relação a câmera

            /*A posição da luz (luz_posicao_cam) calculada acima representa um vetor que sai da origem (0,0,0) e
            aponta para a luz. Para a luz difusa, precisamos calcular um vetor que saia de cada vértice do objeto
            (vertex_posicao_cam) e aponte para essa luz. Para isso, basta calcularmos a diferença entre luz_posicao_cam
            e vertex_posicao_cam.*/
            luz_vetor_cam = luz_posicao_cam - vertex_posicao_cam;//vetor apontando para a luz em relação a posicao do vértice 

            /*Por fim, normalizamos o vetor da luz em relação ao vértice do objeto e calculamos o cosseno do angulo
            entre o mesmo e a normal da superficie utilizando o produto escalar*/
            luz_vetor_cam_normalizada = normalize(luz_vetor_cam);//vetor da luz normalizada
            vertex_normal_cam_normalizada = normalize(vertex_normal_cam);
            float cosseno_difusa = dot(vertex_normal_cam_normalizada,luz_vetor_cam_normalizada);//cosseno do angulo entre o vetor da luz e a normal da superficie
            
            vec3 Id = Ld * Kd * cosseno_difusa;

            return Id;

        }

        vec3 intensidadeEspecular(){
            /*
            Cálculo de Intensidade de Luz Especular (Is)
            Para o cálculo da intensidade de luz especular, precisamos primeiramente calcular o vetor que representa 
            a luz refletida em relação a normal da superfície */
            vec3 luz_reflexao_vetor_cam = reflect(-luz_vetor_cam_normalizada, vertex_normal_cam_normalizada);
            /*Como a intensidade de luz especular depende da posição da câmera, definimos um vetor que sai da superficie
            do objeto e aponta para a camera, e então normalizamos, pois utilizaremos ele no cálculo do produto escalar*/
            vec3 superficie_camera_vetor = normalize(-vertex_posicao_cam);
            /*E então calculamos o ângulo entre o vetor de reflexão da luz e o vetor em relação a posicao do observador*/
            float cosseno_especular = dot(luz_reflexao_vetor_cam, superficie_camera_vetor);
            cosseno_especular = max(cosseno_especular, 0.0);//se o cosseno der negativo, atribui 0 para ele
            /*Na intensidade especular, precisamos elevar o cosseno calculado acima ao expoente especular*/
            float fator_especular = pow (cosseno_especular, especular_exp);
            /*E, por fim, calculamos a intensidade de luz especular refletida (Is) */

            vec3 Is = Ls * Ks * fator_especular;

            return Is;
        }
        void main () {

            vec3 Ia = intensidadeAmbiente();

            vec3 Id = intensidadeDifusa();

            vec3 Is = intensidadeEspecular();

            vec4 corReflexao = vec4(Ia+Id+Is, 1.0);
            vec4 corTextura = texture(s_texture, v_texture);

            /*A cor final do fragmento é a soma das 3 componentes de iluminação + cor da textura*/
            frag_colour = corReflexao + corTextura;
        }
    """
    fs = OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(fs, 512, None)
        print("Erro no fragment shader:\n", info_log)

    # Especificação do Shader Programm:
    Shader_program = OpenGL.GL.shaders.compileProgram(vs, fs)
    if not glGetProgramiv(Shader_program, GL_LINK_STATUS):
        info_log = glGetProgramInfoLog(Shader_program, 512, None)
        print("Erro na linkagem do shader:\n", info_log)

    glDeleteShader(vs)
    glDeleteShader(fs)


def inicializa_shader_luz():
    global shader_luz
    # Especificação do Vertex Shader:
    vertex_shader = """
        #version 400
        layout(location = 0) in vec3 vertex_posicao; //vértices do objeto vindas do modelo do objeto (PYTHON)
        layout(location = 1) in vec2 vertex_texture; //normais do objeto vindas do modelo do objeto (PYTHON)
        layout(location = 2) in vec3 vertex_normal; //normais do objeto vindas do modelo do objeto (PYTHON)
        out vec3 vertex_posicao_cam, vertex_normal_cam;
        out vec2 v_texture; //saida da textura para o fragment shader
        uniform mat4 transform, view, proj;
        void main () {
            vertex_posicao_cam = vec3 (view * transform * vec4 (vertex_posicao, 1.0)); //posição do objeto em relação a CÂMERA
            vertex_normal_cam = vec3 (view *  transform * vec4 (vertex_normal, 0.0)); //normais do objeto em relação a CÂMERA

            v_texture = vertex_texture;

            gl_Position = proj * view * transform * vec4 (vertex_posicao, 1.0);
        }
    """
    vs = OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(vs, 512, None)
        print("Erro no vertex shader:\n", info_log)

    # Especificação do Fragment Shader:
    fragment_shader = """
        #version 400
        in vec3 vertex_posicao_cam, vertex_normal_cam; //variáveis vindas do VERTEX SHADER
        in vec2 v_texture; //variável de textura vinda do VERTEX SHADER
        uniform sampler2D s_texture;

        //propriedades de uma luz pontual vindas do PYTHON
        uniform vec3 luz_posicao;
        uniform vec3 Ls;// luz especular
        uniform vec3 Ld;// luz difusa
        uniform vec3 La;// luz ambiente

        //propriedades de reflexão da superficie do objeto vindas do PYTHON
        uniform vec3 Ks;//reflexão especular
        uniform vec3 Kd;//reflexão difusa
        uniform vec3 Ka;//reflexão ambiente
        uniform float especular_exp;//expoente especular

        uniform mat4 view; //matriz da câmera vinda do PYTHON
        out vec4 frag_colour;

        //variáveis globais que são utilizadas tanto na intensidade difusa quanto especular, para não precisar recalcular duas vezes
        vec3 luz_posicao_cam, luz_vetor_cam, luz_vetor_cam_normalizada, vertex_normal_cam_normalizada;

        vec3 intensidadeAmbiente(){
            /*
            Cálculo de Intensidade de Luz Ambiente (Ia)
            O cálculo da intensidade de luz ambiente é o mais simples:
            basta multiplicar a cor da luz ambiente pela refletância de luz ambiente da superfície
            */
            vec3 Ia = La * Ka;
            return Ia;
        }

        vec3 intensidadeDifusa(){
            /*
            Cálculo de Intensidade de Luz Difusa (Id)
            Para calcularmos a intensidade de luz difusa, precisamos, primeiramente, 
            calcular a posição da luz em relação a câmera (luz_posicao_cam)
            */
            luz_posicao_cam = vec3 (view * vec4 (luz_posicao, 1.0));//posicao da luz em relação a câmera

            /*A posição da luz (luz_posicao_cam) calculada acima representa um vetor que sai da origem (0,0,0) e
            aponta para a luz. Para a luz difusa, precisamos calcular um vetor que saia de cada vértice do objeto
            (vertex_posicao_cam) e aponte para essa luz. Para isso, basta calcularmos a diferença entre luz_posicao_cam
            e vertex_posicao_cam.*/
            luz_vetor_cam = luz_posicao_cam - vertex_posicao_cam;//vetor apontando para a luz em relação a posicao do vértice 

            /*Por fim, normalizamos o vetor da luz em relação ao vértice do objeto e calculamos o cosseno do angulo
            entre o mesmo e a normal da superficie utilizando o produto escalar*/
            luz_vetor_cam_normalizada = normalize(luz_vetor_cam);//vetor da luz normalizada
            vertex_normal_cam_normalizada = normalize(vertex_normal_cam);
            float cosseno_difusa = dot(vertex_normal_cam_normalizada,luz_vetor_cam_normalizada);//cosseno do angulo entre o vetor da luz e a normal da superficie

            vec3 Id = Ld * Kd * cosseno_difusa;

            return Id;

        }

        vec3 intensidadeEspecular(){
            /*
            Cálculo de Intensidade de Luz Especular (Is)
            Para o cálculo da intensidade de luz especular, precisamos primeiramente calcular o vetor que representa 
            a luz refletida em relação a normal da superfície */
            vec3 luz_reflexao_vetor_cam = reflect(-luz_vetor_cam_normalizada, vertex_normal_cam_normalizada);
            /*Como a intensidade de luz especular depende da posição da câmera, definimos um vetor que sai da superficie
            do objeto e aponta para a camera, e então normalizamos, pois utilizaremos ele no cálculo do produto escalar*/
            vec3 superficie_camera_vetor = normalize(-vertex_posicao_cam);
            /*E então calculamos o ângulo entre o vetor de reflexão da luz e o vetor em relação a posicao do observador*/
            float cosseno_especular = dot(luz_reflexao_vetor_cam, superficie_camera_vetor);
            cosseno_especular = max(cosseno_especular, 0.0);//se o cosseno der negativo, atribui 0 para ele
            /*Na intensidade especular, precisamos elevar o cosseno calculado acima ao expoente especular*/
            float fator_especular = pow (cosseno_especular, especular_exp);
            /*E, por fim, calculamos a intensidade de luz especular refletida (Is) */

            vec3 Is = Ls * Ks * fator_especular;

            return Is;
        }
        void main () {

            vec3 Ia = intensidadeAmbiente();

            vec3 Id = intensidadeDifusa();

            vec3 Is = intensidadeEspecular();

            vec4 corReflexao = vec4(Ia+Id+Is, 1.0);
            vec4 corTextura = texture(s_texture, v_texture);

            /*A cor final do fragmento é a soma das 3 componentes de iluminação + cor da textura*/
            frag_colour = corReflexao + corTextura;
        }
    """
    fs = OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(fs, 512, None)
        print("Erro no fragment shader:\n", info_log)

    # Especificação do Shader Programm:
    shader_luz = OpenGL.GL.shaders.compileProgram(vs, fs)
    if not glGetProgramiv(shader_luz, GL_LINK_STATUS):
        info_log = glGetProgramInfoLog(shader_luz, 512, None)
        print("Erro na linkagem do shader:\n", info_log)

    glDeleteShader(vs)
    glDeleteShader(fs)


def transformacaoGenerica(lista_valores):
    Tx, Ty, Tz = lista_valores[0:3]
    Sx, Sy, Sz = lista_valores[3:6]
    Rx, Ry, Rz = lista_valores[6:9]
    # matriz de translação
    translacao = pyrr.matrix44.create_from_translation(pyrr.Vector3([Tx, Ty, Tz]))

    # matriz de rotação em torno do eixo X
    anguloX = np.radians(Rx)
    rotacaoX = pyrr.matrix44.create_from_x_rotation(anguloX)

    # matriz de rotação em torno do eixo Y
    anguloY = np.radians(Ry)
    rotacaoY = pyrr.matrix44.create_from_y_rotation(anguloY)

    # matriz de rotação em torno do eixo Z
    anguloZ = np.radians(Rz)
    rotacaoZ = pyrr.matrix44.create_from_z_rotation(anguloZ)

    # combinação das 3 rotação
    rotacao = pyrr.matrix44.multiply(rotacaoY, rotacaoX)
    rotacao = pyrr.matrix44.multiply(rotacaoZ, rotacao)

    # matriz de escala
    escala = pyrr.matrix44.create_from_scale(pyrr.Vector3([Sx, Sy, Sz]))

    transformacaoFinal = pyrr.matrix44.multiply(rotacao, escala)
    transformacaoFinal = pyrr.matrix44.multiply(translacao, transformacaoFinal)

    # E passamos a matriz para o Vertex Shader.
    transformLoc = glGetUniformLocation(Shader_program, "transform")
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transformacaoFinal)


def especificaMatrizVisualizacao():
    visualizacao = cam.get_view_matrix()
    transformLoc = glGetUniformLocation(Shader_program, "view")
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, visualizacao)


def especificaMatrizProjecao():
    # Especificação da matriz de projeção perspectiva.
    projecao = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 100)
    transformLoc = glGetUniformLocation(Shader_program, "proj")
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, projecao)


def inicializaCamera():
    especificaMatrizVisualizacao()
    especificaMatrizProjecao()


def especificaMaterial(KaR, KaG, KaB, KdR, KdG, KdB, KsR, KsG, KsB, n):
    global Shader_program
    # Coeficiente de reflexão ambiente
    Ka = pyrr.Vector3([KaR, KaG, KaB])  # reflete luz ambiente
    Ka_loc = glGetUniformLocation(Shader_program, "Ka")
    glUniform3fv(Ka_loc, 1, Ka)

    # Coeficiente de reflexão difusa
    Kd = pyrr.Vector3([KdR, KdG, KdB])  # reflete luz difusa
    Kd_loc = glGetUniformLocation(Shader_program, "Kd")
    glUniform3fv(Kd_loc, 1, Kd)

    # Coeficiente de reflexão especular
    Ks = pyrr.Vector3([KsR, KsG, KsB])  # reflete luz especular
    Ks_loc = glGetUniformLocation(Shader_program, "Ks")
    glUniform3fv(Ks_loc, 1, Ks)

    # expoente expecular
    especular_exp = n
    especular_exp_loc = glGetUniformLocation(Shader_program, "especular_exp")
    glUniform1f(especular_exp_loc, especular_exp)


def especificaLuz():
    global Shader_program, luz_posicao, La, Ld, Ls
    # posição da luz
    luz_posicaoloc = glGetUniformLocation(Shader_program,
                                          "luz_posicao")  # envia o array da posição da luz para o shader
    glUniform3fv(luz_posicaoloc, 1, luz_posicao)

    # Fonte de luz ambiente
    La_loc = glGetUniformLocation(Shader_program, "La")  # envia o array da Luz Ambiente para o shader
    glUniform3fv(La_loc, 1, La)

    # Fonte de luz difusa
    Ld_loc = glGetUniformLocation(Shader_program, "Ld")  # envia o array da Luz Difusa para o shader
    glUniform3fv(Ld_loc, 1, Ld)

    # Fonte de luz especular
    Ls_loc = glGetUniformLocation(Shader_program, "Ls")  # envia o array da Luz Especular para o shader
    glUniform3fv(Ls_loc, 1, Ls)


def especifica_luz_2():
    global shader_luz, luz_2_posicao, La2, Ld2, Ls2
    # posição da luz
    luz_2_posicaoloc = glGetUniformLocation(shader_luz,
                                            "luz_posicao")  # envia o array da posição da luz para o shader
    glUniform3fv(luz_2_posicaoloc, 1, luz_2_posicao)

    # Fonte de luz ambiente
    La_2_loc = glGetUniformLocation(shader_luz, "La")  # envia o array da Luz Ambiente para o shader
    glUniform3fv(La_2_loc, 1, La2)

    # Fonte de luz difusa
    Ld_2_loc = glGetUniformLocation(shader_luz, "Ld")  # envia o array da Luz Difusa para o shader
    glUniform3fv(Ld_2_loc, 1, Ld2)

    # Fonte de luz especular
    Ls_2_loc = glGetUniformLocation(shader_luz, "Ls")  # envia o array da Luz Especular para o shader
    glUniform3fv(Ls_2_loc, 1, Ls2)

def inicializaRenderizacao():
    global Window, Shader_program, vao_retangulo, shader_luz, WIDTH, HEIGHT, Tempo_entre_frames

    tempo_anterior = glfw.get_time()

    # Ativação do teste de profundidade. Sem ele, o OpenGL não sabe que faces devem ficar na frente e que faces devem
    # ficar atrás.
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    while not glfw.window_should_close(Window):
        # calcula quantos segundos se passaram entre um frame e outro
        tempo_frame_atual = glfw.get_time()
        Tempo_entre_frames = tempo_frame_atual - tempo_anterior
        tempo_anterior = tempo_frame_atual

        movimentaCamera()

        # limpa a tela e os buffers
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # configura a viewport para pegar toda a janela
        glViewport(0, 0, WIDTH, HEIGHT)

        glUseProgram(shader_luz)
        especifica_luz_2()

        # ativa o shader
        glUseProgram(Shader_program)

        especificaLuz()  # parâmetros da fonte de luz

        inicializaCamera()  # configuramos a câmera

        especificaMaterial(0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 32)

        glBindVertexArray(vao_retangulo)
        transformacaoGenerica([0, -0.6, 0, 100, 1, 100, 0, 0,0])
        glDrawArrays(GL_TRIANGLES, 0, 36)

        for objeto in objetos:
            glBindVertexArray(objeto['vao_objeto'])  # ativamos o objeto que queremos renderizar

            glBindTexture(GL_TEXTURE_2D, objeto['obj_textura'])
            # Set the texture wrapping parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            # Set texture filtering parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, objeto['tex_w'], objeto['tex_h'], 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         objeto['tex_data'])

            transformacaoGenerica(objeto['transformacao'])
            glDrawArrays(GL_TRIANGLES, 0, len(objeto['obj_indices']))

        glfw.poll_events()

        glfw.swap_buffers(Window)

    glfw.terminate()


inicializaOpenGL()
inicializaObjetos()
inicializa_retangulo()
inicializa_shader_luz()
inicializa_shaders()
inicializaRenderizacao()
