
class Isoredshift:
    def __init__(
            self, 
            inclination, 
            redshift, 
            bh_mass, 
            solver_parameters=None, 
            from_isoradials=None
    ):
        # Parent black hole parameters
        if from_isoradials is None:
            from_isoradials = {}

        self.t = inclination
        self.M = bh_mass
        self.t = inclination
        self.redshift = redshift

        # Parent isoradial(s) solver parameters: recycled here.
        # TODO: currently same as Isoradial out of laziness, but these might require different solver params
        self.solver_params = solver_parameters if solver_parameters else \
            {'initial_guesses': 20,
             'midpoint_iterations': 10,
             'plot_inbetween': False,
             'min_periastron': 3.001 * self.M,
             'retry_angular_precision': 30,
             "retry_tip": 15}

        # Isoredshift attributes
        self.radii_w_coordinates_dict = {}
        if from_isoradials is not None:
            self.calc_from_isoradials(from_isoradials)
        else:
            pass  # TODO: initialise from photon sphere isoradial?
        self.coordinates_with_radii_dict = self.__init_co_to_radii_dict()
        self.ir_radii_w_co = [key for key, val in self.radii_w_coordinates_dict.items() if
                              len(val[0]) > 0]  # list of R that have solution
        self.co = self.angles, self.radii = self.__extract_co_from_solutions_dict()
        self.max_radius = max(self.radii) if len(self.radii) else 0
        self.x, self.y = transform.polar_to_cartesian_lists(self.radii, self.angles, rotation=0)
        self.order_coordinates()

    def __update(self):
        self.ir_radii_w_co = [key for key, val in self.radii_w_coordinates_dict.items() if
                              len(val[0]) > 0]  # list of R that have solution
        self.co = self.angles, self.radii = self.__extract_co_from_solutions_dict()
        self.x, self.y = transform.polar_to_cartesian_lists(self.radii, self.angles, rotation=0)
        self.order_coordinates()

    def __add_solutions(self, angles, impact_parameters, radius_ir):
        def __add_solution(__iz: Isoredshift, __angle, __radius_b, __radius_ir):
            """
            Updates all attributes to contain newly found solution
            :return:
            """
            if __radius_ir in __iz.radii_w_coordinates_dict:  # radius is already considered
                if len(__iz.radii_w_coordinates_dict[__radius_ir][0]):  # radius already has a solution
                    __iz.radii_w_coordinates_dict[__radius_ir][0].append(__angle)
                    __iz.radii_w_coordinates_dict[__radius_ir][1].append(__radius_b)
                else:
                    __iz.radii_w_coordinates_dict[__radius_ir] = [[__angle], [__radius_b]]
            else:
                __iz.radii_w_coordinates_dict[__radius_ir] = [[__angle], [__radius_b]]
            __iz.coordinates_with_radii_dict[(__angle, __radius_b)] = __radius_ir
            __iz.__update()

        for angle, impact_parameter in zip(angles, impact_parameters):
            __add_solution(self, angle, impact_parameter, radius_ir)

    def __init_co_to_radii_dict(self):
        to_return = {}
        for radius, co in self.radii_w_coordinates_dict.items():
            if len(co[0]):  # if radius has solution
                coordinates = [tuple(e) for e in np.array(co).T]  # TODO do these need to be lists actually?
                for co_ in coordinates:  # either one or two solutions
                    to_return[co_] = radius
        return to_return

    def __extract_co_from_solutions_dict(self):
        a = []
        r = []
        for key, val in self.radii_w_coordinates_dict.items():
            if len(val[0]) > 0:  # at least one solution was found
                angles, radii = val
                [a.append(angle) for angle in angles]
                [r.append(radius) for radius in radii]
        self.co = self.angles, self.radii = a, r
        return a, r

    def calc_from_isoradials(self, isoradials, cartesian=False):
        """
        Calculates the isoredshift for a single redshift value, based on a couple of isoradials calculated
        at low precision
        """
        solutions = OrderedDict()
        _max_radius = None
        for ir in isoradials:
            # Use the same solver params from the black hole to calculate the redshift location on the isoradial
            a, r = ir.calc_redshift_location_on_ir(self.redshift, cartesian=cartesian)
            solutions[ir.radius] = [a, r]
        self.radii_w_coordinates_dict = solutions
        self.__update()

    def split_co_on_solutions(self):
        """
        Iterates the dictionary of coordinates that looks like {r_0: [[angle1, angle2], [b_1, b_2]],
        r_1: [[...], [...]]}
        Checks if each key (radius corresponding to an isoradial) has solutions for the isoredshift or not.
        Splits the original dict in two: one with solutions and one without solutions

        :returns: two dictionaries: one with solutions and one without.
        """
        keys_w_s = []
        keys_wo_s = []
        for key in self.radii_w_coordinates_dict:
            if len(self.radii_w_coordinates_dict[key][0]) == 0:
                keys_wo_s.append(key)
            else:
                keys_w_s.append(key)
        dict_w_s = {key: self.radii_w_coordinates_dict[key] for key in keys_w_s}
        dict_wo_s = {key: self.radii_w_coordinates_dict[key] for key in keys_wo_s}
        return dict_w_s, dict_wo_s

    def calc_core_coordinates(self):
        """Calculates the coordinates of the redshift on the closest possible isoradial: 6*M (= 2*R_s)"""
        ir = Isoradial(6. * self.M, self.t, self.M, order=0, **self.solver_params)
        co = ir.calc_redshift_location_on_ir(self.redshift)
        return co

    def order_coordinates(self, plot_title="", plot_inbetween=False):
        angles, radii = self.co
        co = [(a, r) for a, r in zip(angles, radii)]
        x, y = transform.polar_to_cartesian_lists(radii, angles)
        cx, cy = np.mean(x, axis=0), np.mean(y, axis=0)
        order_around = [.3 * cx, .8 * cy]

        sorted_co = sorted(
            co, key=lambda polar_point: get_angle_around(
                order_around, transform.polar_to_cartesian_single(polar_point[0], polar_point[1]))
        )

        if plot_inbetween:
            # use this to get a visual overview of what happens when ordering the isoradial points using
            # getAngleAround() as a key
            fig, ax = plt.subplots()
            for i, p in enumerate(sorted_co):
                plt.plot(*transform.polar_to_cartesian_single(*p), 'bo')
                plt.text(x[i] * (1 + 0.01), y[i] * (1 + 0.01), i, fontsize=12)
            plt.plot(*np.array([transform.polar_to_cartesian_single(*p) for p in sorted_co]).T)
            plt.scatter(*order_around)
            plt.plot([0, order_around[0]], [0, order_around[1]])
            plt.title(plot_title)
            plt.show()
            plt.close('all')

        self.co = self.angles, self.radii = [e[0] for e in sorted_co], [e[1] for e in sorted_co]
        self.x, self.y = transform.polar_to_cartesian_lists(self.radii, self.angles, rotation=0)

    def calc_redshift_on_ir_between_angles(
            self, 
            radius, 
            begin_angle=0, 
            end_angle=np.pi, 
            angular_precision=3,
            mirror=False, 
            plot_inbetween=False, 
            title='', 
            force_solution=False):
        ir = Isoradial(radius=radius, incl=self.t, mass=self.M,
                       angular_properties={'start_angle': begin_angle,
                                           'end_angle': end_angle,
                                           'angular_precision': angular_precision,
                                           'mirror': mirror})
        ir.find_redshift_params['force_redshift_solution'] = force_solution
        a, r = ir.calc_redshift_location_on_ir(self.redshift, cartesian=False)
        if plot_inbetween:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.axhline(self.redshift)
            fig.suptitle(title)
            ir.plot_redshift(fig, ax, show=False)
            fig.savefig('Plots/{}.png'.format(title))
        return a, r

    def improve_between_all_solutions_once(self):
        """
        Calculates the redshift on the isoredshift line between the already known redshifts
        Does so by calculating the entire isoradial (with low precision) inbetween the radii corresponding
        to these redshift solutions and calculating the redshifts on this isoradial
        """

        self.order_coordinates()  # TODO: is this necessary or already done before? currently depends on further implementation
        co = [(angle, radius_b) for angle, radius_b in zip(*self.co)]
        i = 0
        for b, e in zip(co[:-1], co[1:]):
            r_inbetw = .5 * (self.coordinates_with_radii_dict[b] + self.coordinates_with_radii_dict[e])
            begin_angle, end_angle = b[0], e[0]
            if end_angle - begin_angle > np.pi:
                begin_angle, end_angle = end_angle, begin_angle
            # calc location of redshift, guaranteed to exist between the angles begin_angle and end_angle
            # NOT guaranteed to exist at r_inbetw (isoradial radius, not impact parameter):
            #   1. If coordinates aren't split on a jump (either on the black hole or if they never meet at inf)
            #   2. If we're trying to find one at a tip -> should be covered by other methods though.
            a, r = self.calc_redshift_on_ir_between_angles(r_inbetw, begin_angle - .1, end_angle + .1,
                                                           plot_inbetween=False,
                                                           title='between p{} and p{}'.format(i, i + 1),
                                                           force_solution=True)
            i += 1
            if len(a):
                self.__add_solutions(a, r, r_inbetw)

    def recalc_redshift_on_closest_isoradial_wo_z(self):
        """
        Recalculates the first (closest) isoradial that did not find a solution with more angular precision.
        Isoradial is recalculated withing the angular interval of the two last (furthest) solutions.
        This is done to guarantee that the lack of solutions is not due to lack of angular precision.

        :return: 0 if success
        """

        r_w_s, r_wo_s = self.split_co_on_solutions()
        angle_interval, last_radii = self.radii_w_coordinates_dict[max(r_w_s.keys())]
        assert len(angle_interval) > 1, "1 or less angles found for corresponding isoradial R={}".format(max(r_w_s))
        closest_r_wo_s = min(r_wo_s.keys())
        begin_angle, end_angle = angle_interval
        if end_angle - begin_angle > np.pi:  # in case the angle is around 0 and 2pi
            begin_angle, end_angle = end_angle, begin_angle  # this works, apparently
        # calculate solutions and add them to the class attributes if they exist
        a, b = self.calc_redshift_on_ir_between_angles(
            closest_r_wo_s, 
            begin_angle, end_angle,
            angular_precision=
            self.solver_params['retry_angular_precision'],
            mirror=False)
        if len(a):
            self.__add_solutions(a, b, closest_r_wo_s)
        return a, b

    def recalc_isoradials_wo_redshift_solutions(self, plot_inbetween=False):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s.keys()) > 0 and len(r_w_so) > 0:
            a, r = self.recalc_redshift_on_closest_isoradial_wo_z()  # re-calculate isoradials where no solutions were found
            self.order_coordinates(plot_title="improving tip angular")
            r_w_so, r_wo_s = self.split_co_on_solutions()
            while len(a) > 0 and len(r_wo_s.keys()) > 0:
                a, r = self.recalc_redshift_on_closest_isoradial_wo_z()  # re-calculate isoradials where no solutions were found
                r_w_s, r_wo_s = self.split_co_on_solutions()
                self.order_coordinates(plot_inbetween=plot_inbetween, plot_title="improving tip angular")

    def calc_ir_before_closest_ir_wo_z(self, angular_margin=.3):
        """
        Given two isoradials (one with solutions and one without), calculates a new isoradial inbetween the two.
        Either a solution is found, or the location of the tip of the isoredshift is more closed in.
        """
        r_w_s, r_wo_s = self.split_co_on_solutions()
        angle_interval, last_radii = self.radii_w_coordinates_dict[max(r_w_s.keys())]  # one isoradial: two angles/radii
        if len(r_wo_s.keys()) > 0 and len(r_w_s) > 0:  # assert there are radii considered without solutions
            first_r_wo_s = min(r_wo_s.keys())
            last_r_w_s = max(r_w_s.keys())
            inbetween_r = .5 * (first_r_wo_s + last_r_w_s)
            begin_angle, end_angle = angle_interval
            if end_angle - begin_angle > np.pi:  # in case the angle is around 0 and 2pi
                begin_angle, end_angle = end_angle, begin_angle  # this works, apparently
            a, r = self.calc_redshift_on_ir_between_angles(
                inbetween_r, 
                begin_angle - angular_margin,
                end_angle + angular_margin,
                angular_precision=
                self.solver_params['retry_angular_precision'],
                mirror=False)
            if len(a):
                self.__add_solutions(a, r, inbetween_r)
            else:
                self.radii_w_coordinates_dict[inbetween_r] = [[], []]

    def improve_tip(self, iterations=6):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s.keys()) > 0:
            for it in range(iterations):
                self.calc_ir_before_closest_ir_wo_z()
                self.order_coordinates(plot_title=f"Improving tip iteration {it}",
                                       plot_inbetween=self.solver_params["plot_inbetween"])

    def improve(self):
        """
        Given an isoredshift calculated from just a couple coordinates, improves the solutions by:
        1. recalculating isoradials that did not contain the wanted redshift with more precision
        2. calculating isoradials inbetween the largest isoradial that had the wanted redshift and
        the closest that did not.
        """
        r_w_s, r_wo_s = self.split_co_on_solutions()
        if len(r_w_s):  # at least one solution is found
            self.recalc_isoradials_wo_redshift_solutions(plot_inbetween=False)
            self.improve_tip(iterations=self.solver_params["retry_tip"])
            for n in range(self.solver_params["times_inbetween"]):
                self.improve_between_all_solutions_once()
                self.order_coordinates(plot_title="calculating inbetween",
                                       plot_inbetween=self.solver_params["plot_inbetween"])

    def split_co_on_jump(self, threshold=2):
        """
        Returns the index where the difference in isoredshift coordinate values is significantly bigger than the median
        distance. This is used to avoid the plotter to connect two points that should not be connected.
        A jump between two coordinates co1 and co2 is when the isoredshift line does not connect within
        the considered frame, but either does not connect (purely radial isoredshift lines), or connects very far
        from the black hole
        """

        def dist(__x, __y):
            x1, x2 = __x
            y1, y2 = __y
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return d

        self.order_coordinates()
        self.__update()
        x, y = transform.polar_to_cartesian_lists(self.radii, self.angles)
        _dist = [dist((x1, x2), (y1, y2)) for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:])]
        mx2, mx = sorted(_dist)[-2:]
        if mx > threshold * mx2:
            split_ind = np.where(_dist == mx)[0][0]
            if not abs(np.diff(np.sign(self.x[split_ind:split_ind + 2]))) > 0:
                # not really a jump, just an artefact of varying point density along the isoredshift line
                split_ind = None
        else:
            split_ind = None
        return split_ind

    def plot(self, norm, color_map):
        color = cm.ScalarMappable(norm=norm, cmap=color_map).to_rgba(self.redshift)
        plt.plot(self.y, [-e for e in self.x], color=color)  # TODO: hack to correctly orient plot
        plt.plot(self.y, [-e for e in self.x], color=color)  # TODO: hack to correctly orient plot
        tries = 0
        while len(self.ir_radii_w_co) < 10 and tries < 10:
            self.improve_between_all_solutions_once()
            tries += 1

        plt.plot(self.y, [-e for e in self.x], color=color)  # TODO: hack to correctly orient plot
